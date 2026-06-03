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
    help="Comma-separated bet types for combined mode (e.g. 'mot_so,cong_tong,lon_hoa_nho')",
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
        vietlott-bingo18 play --budget 10000000 --mode combine --bet-types mot_so,cong_tong,lon_hoa_nho

        # All-in mode: best bet only
        vietlott-bingo18 play --budget 10000000 --mode all_in --bet-types cong_tong,lon_hoa_nho

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

    # Export per-agent bet decisions to CSV
    from machine_learning.bingo18.visualize import export_agent_decisions_csv

    agent_csv_dicts = [{"agent_id": ar.agent_id, "bet_history": ar.bet_history} for ar in result.agent_results]
    export_agent_decisions_csv(agent_csv_dicts, output_dir)

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

    all_agents = sorted(result.agent_results, key=lambda a: a.roi, reverse=True)
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
@click.option(
    "--output",
    type=click.Path(),
    default="models/bingo18/parallel",
    help="Output directory for parallel training models",
)
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
        object.__setattr__(config, "n_epochs", epochs)
        object.__setattr__(config, "budget", budget)

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
    click.echo(f"  {'-' * 30} {'-' * 10} {'-' * 10} {'-' * 15} {'-' * 8} {'-' * 8}")

    for r in sorted(result.all_results, key=lambda x: x.test_roi, reverse=True):
        click.echo(
            f"  {r.config.name:<30} {r.test_roi:>9.1f}% {r.test_win_rate:>9.1%} "
            f"{r.test_final_budget:>15,} {r.test_total_bets:>8} {r.elapsed_seconds:>7.1f}s"
        )

    click.echo(f"\n  BEST: {result.best.config.name}")
    click.echo(f"  Model saved: {result.best.model_path}")
    click.echo(f"  Total time: {result.total_elapsed:.1f}s")
    click.echo(f"{'=' * 80}")


@cli.command()
@click.option("--n-agents", type=int, default=6, help="Number of agents to train in parallel")
@click.option("--rounds", type=int, default=3, help="Number of full passes through data")
@click.option("--budget", type=int, default=500_000, help="Starting budget per agent")
@click.option("--bet-size", type=int, default=10_000, help="Base bet size")
@click.option("--adapt-interval", type=int, default=50, help="Draws between adaptation cycles")
@click.option(
    "--agent-dir", type=click.Path(), default="models/bingo18/agents", help="Directory to save/load agent state"
)
@click.option("--data-path", type=click.Path(exists=True), default=None, help="Path to bingo18.jsonl")
@click.option("--fresh", is_flag=True, help="Ignore existing agent state, start fresh")
@click.option(
    "--log-file", type=click.Path(), default=None, help="Write logs to file (default: logs/bingo18_YYYYMMDD_HHMMSS.log)"
)
def train_live(
    n_agents: int,
    rounds: int,
    budget: int,
    bet_size: int,
    adapt_interval: int,
    agent_dir: str,
    data_path: str | None,
    fresh: bool,
    log_file: str | None,
):
    """Train multiple agents in parallel with real-time output.

    Agents learn persistently, self-heal on bankruptcy, and save state
    to disk. Re-run to continue training from where agents left off.

    Examples:

        # Default: 6 agents, 3 rounds
        vietlott-bingo18 train-live

        # More agents, more rounds
        vietlott-bingo18 train-live --n-agents 10 --rounds 10

        # Continue from saved state
        vietlott-bingo18 train-live --rounds 5

        # Start fresh (ignore saved state)
        vietlott-bingo18 train-live --fresh --rounds 5
    """
    import time as _time

    from machine_learning.bingo18.continuous_trainer import run_continuous_training

    # Set up file logging
    log_path = Path(log_file) if log_file else Path("logs") / f"bingo18_{_time.strftime('%Y%m%d_%H%M%S')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_id = logger.add(
        log_path, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}", level="INFO", encoding="utf-8"
    )
    logger.info(f"Logging to {log_path}")
    click.echo(f"  Log file: {log_path}")

    # Load data
    data_path_obj = Path(data_path) if data_path else DEFAULT_DATA_PATH
    df = load_data(data_path_obj)

    # Load or train Stage 1 model
    model_path = Path(agent_dir) / "stage1_model.joblib"
    model = Bingo18Model()
    if model_path.exists() and not fresh:
        logger.info(f"Loading Stage 1 model from {model_path}...")
        model.load(model_path)
        logger.info("Stage 1 model loaded.")
    else:
        logger.info("Training Stage 1 model (digit prediction)...")
        metrics = model.train(df)
        logger.info(f"Stage 1 model trained: {metrics.summary()}")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)

    # Run continuous training
    result = run_continuous_training(
        model=model,
        df=df,
        n_agents=n_agents,
        n_rounds=rounds,
        budget=budget,
        bet_size=bet_size,
        adaptation_interval=adapt_interval,
        agent_state_dir=agent_dir,
        load_existing=not fresh,
    )

    logger.remove(log_id)

    # Print final summary
    click.echo(f"\n{'=' * 80}")
    click.echo("  CONTINUOUS TRAINING COMPLETE")
    click.echo(f"{'=' * 80}")
    click.echo(f"  Agents: {result.n_agents}")
    click.echo(f"  Rounds: {result.total_rounds}")
    click.echo(f"  Time: {result.total_elapsed_seconds:.1f}s")
    click.echo()
    click.echo(f"  {'Agent':<15} {'ROI':>8} {'Budget':>12} {'Win%':>7} {'Bets':>7} {'Bankrupt':>9} {'Status':>7}")
    click.echo(f"  {'-' * 15} {'-' * 8} {'-' * 12} {'-' * 7} {'-' * 7} {'-' * 9} {'-' * 7}")
    for r in result.agent_results:
        status = "ALIVE" if r.survived else "DEAD"
        click.echo(
            f"  {r.agent_id:<15} {r.roi:>+7.1f}% {r.final_budget:>11,} "
            f"{r.win_rate:>6.1%} {r.total_bets:>6,} {r.bankruptcies:>8} {status:>7}"
        )
    click.echo(f"{'=' * 80}")


@cli.command()
@click.option(
    "--agent-dir", type=click.Path(), default="models/bingo18/agents", help="Directory with saved agent state"
)
@click.option("--top-bets", type=int, default=3, help="Number of top bet types to show per agent (default: 3)")
@click.option("--detail", is_flag=True, help="Show full bet type breakdown per agent")
def stats(agent_dir: str, top_bets: int, detail: bool):
    """Show statistics for all saved agents.

    Reads agent JSON files and prints a leaderboard + per-agent bet type performance.

    Examples:

        # Quick overview
        vietlott-bingo18 stats

        # Show top 5 bet types per agent
        vietlott-bingo18 stats --top-bets 5

        # Full bet type breakdown
        vietlott-bingo18 stats --detail
    """
    import json

    state_dir = Path(agent_dir)
    files = sorted(state_dir.glob("*.json"))

    if not files:
        click.echo(f"No agent files found in {state_dir}")
        return

    agents_data = []
    for filepath in files:
        try:
            with filepath.open("r", encoding="utf-8") as f:
                data = json.load(f)
            agents_data.append(data)
        except Exception as e:
            click.echo(f"  [WARN] Failed to load {filepath.name}: {e}")

    if not agents_data:
        click.echo("No valid agent files found.")
        return

    # Compute derived stats for each agent
    rows = []
    for d in agents_data:
        s = d.get("state", {})
        budget = d.get("budget", 0)
        starting = s.get("starting_budget", d.get("budget", 500_000))
        total_bets = s.get("total_bets", 0)
        wins = s.get("wins", 0)
        generation = s.get("generation", 0)
        win_rate = wins / total_bets if total_bets > 0 else 0.0
        roi = (budget - starting) / starting * 100 if starting > 0 else 0.0

        # Lifetime ROI from bet_type_stats — never resets on bankruptcy, most reliable metric
        bet_stats = s.get("bet_type_stats", {})
        life_wagered = sum(bs.get("total_wagered", 0) for bs in bet_stats.values())
        life_payout = sum(bs.get("total_payout", 0) for bs in bet_stats.values())
        lifetime_roi = (life_payout - life_wagered) / life_wagered * 100 if life_wagered > 0 else 0.0

        # Per-bet-type performance
        bet_perf = []
        for bt_name, bs in bet_stats.items():
            wagered = bs.get("total_wagered", 0)
            payout = bs.get("total_payout", 0)
            bets = bs.get("total_bets", 0)
            bt_wins = bs.get("wins", 0)
            if wagered > 0:
                bt_roi = (payout - wagered) / wagered * 100
                bt_wr = bt_wins / bets if bets > 0 else 0.0
                bet_perf.append((bt_name, bt_roi, bt_wr, bets))

        bet_perf.sort(key=lambda x: x[1], reverse=True)

        rows.append(
            {
                "agent_id": d.get("agent_id", "?"),
                "genome": d.get("genome", {}),
                "budget": budget,
                "starting_budget": starting,
                "roi": roi,
                "lifetime_roi": lifetime_roi,
                "win_rate": win_rate,
                "total_bets": total_bets,
                "generation": generation,
                "bet_perf": bet_perf,
                "bet_weights": s.get("bet_type_weights", {}),
            }
        )

    # Sort by lifetime ROI — more reliable than snapshot ROI
    rows.sort(key=lambda r: r["lifetime_roi"], reverse=True)

    # --- Leaderboard ---
    click.echo(f"\n{'=' * 115}")
    click.echo(f"  AGENT STATISTICS  ({len(rows)} agents from {state_dir})  [sorted by Lifetime ROI]")
    click.echo(f"{'=' * 115}")
    click.echo(
        f"  {'Agent':<15} {'Gen':>5} {'Budget':>12} {'SnapshotROI':>12} {'LifetimeROI':>12} "
        f"{'WinRate':>8} {'Bets':>7} {'Risk':>12} {'Strategy':>10}  Best Bet Types"
    )
    click.echo(f"  {'-' * 110}")

    for r in rows:
        g = r["genome"]
        top = r["bet_perf"][:top_bets]
        top_str = "  ".join(f"{bt}({roi:+.0f}%)" for bt, roi, _, _ in top) if top else "-"
        snap_str = f"{r['roi']:>+10.1f}%"
        life_str = f"{r['lifetime_roi']:>+10.1f}%"
        click.echo(
            f"  {r['agent_id']:<15} {r['generation']:>5} {r['budget']:>12,} "
            f"{snap_str:>12} {life_str:>12} "
            f"{r['win_rate']:>7.1%} {r['total_bets']:>7,} "
            f"{g.get('risk_profile', '?'):>12} {g.get('primary_strategy', '?'):>10}  {top_str}"
        )

    click.echo(f"{'=' * 115}")

    # --- Aggregate summary ---
    total_bets_all = sum(r["total_bets"] for r in rows)
    alive = sum(1 for r in rows if r["budget"] >= 10_000)
    avg_lifetime = sum(r["lifetime_roi"] for r in rows) / len(rows)
    avg_snap = sum(r["roi"] for r in rows) / len(rows)
    best = rows[0]
    click.echo(
        f"\n  Summary: {alive}/{len(rows)} alive | "
        f"avg LifetimeROI {avg_lifetime:+.1f}% | avg SnapshotROI {avg_snap:+.1f}% | "
        f"total bets {total_bets_all:,} | best: {best['agent_id']} (lifetime {best['lifetime_roi']:+.1f}%)"
    )

    # --- Aggregate bet type performance ---
    click.echo(f"\n  BET TYPE PERFORMANCE (aggregate across all agents)")
    click.echo(f"  {'-' * 60}")
    agg: dict[str, dict] = {}
    for d in agents_data:
        for bt_name, bs in d.get("state", {}).get("bet_type_stats", {}).items():
            if bt_name not in agg:
                agg[bt_name] = {"wagered": 0, "payout": 0, "bets": 0, "wins": 0}
            agg[bt_name]["wagered"] += bs.get("total_wagered", 0)
            agg[bt_name]["payout"] += bs.get("total_payout", 0)
            agg[bt_name]["bets"] += bs.get("total_bets", 0)
            agg[bt_name]["wins"] += bs.get("wins", 0)

    agg_rows = []
    for bt_name, v in agg.items():
        if v["wagered"] > 0:
            roi = (v["payout"] - v["wagered"]) / v["wagered"] * 100
            wr = v["wins"] / v["bets"] if v["bets"] > 0 else 0.0
            agg_rows.append((bt_name, roi, wr, v["bets"]))
    agg_rows.sort(key=lambda x: x[1], reverse=True)

    click.echo(f"  {'Bet Type':<20} {'ROI':>8} {'WinRate':>8} {'Total Bets':>12}")
    click.echo(f"  {'-' * 55}")
    for bt_name, roi, wr, bets in agg_rows:
        roi_str = f"{roi:>+7.1f}%"
        click.echo(f"  {bt_name:<20} {roi_str:>8} {wr:>7.1%} {bets:>12,}")

    # --- Detail per agent ---
    if detail:
        for r in rows:
            click.echo(
                f"\n  [{r['agent_id']}]  gen={r['generation']}  "
                f"SnapshotROI={r['roi']:+.1f}%  LifetimeROI={r['lifetime_roi']:+.1f}%  "
                f"budget={r['budget']:,}  bets={r['total_bets']:,}"
            )
            click.echo(f"    {'Bet Type':<20} {'Weight':>7} {'ROI':>8} {'WinRate':>8} {'Bets':>8}")
            weights = r["bet_weights"]
            for bt_name, bt_roi, bt_wr, bets in r["bet_perf"]:
                w = weights.get(bt_name, 1.0)
                roi_s = f"{bt_roi:>+7.1f}%"
                click.echo(f"    {bt_name:<20} {w:>7.2f} {roi_s:>8} {bt_wr:>7.1%} {bets:>8,}")


@cli.command()
@click.option(
    "--agent-dir", type=click.Path(), default="models/bingo18/agents", help="Directory with saved agent states"
)
@click.option(
    "--data-path", type=click.Path(exists=True), default=None, help="New data to evaluate on (default: bingo18.jsonl)"
)
@click.option("--top-n", type=int, default=3, help="Evaluate top N agents by saved ROI (default: 3, 0 = all)")
@click.option("--agent", type=str, default=None, help="Evaluate a specific agent by ID (e.g. agent_000)")
def eval(agent_dir: str, data_path: str | None, top_n: int, agent: str | None):
    """Evaluate saved agents on new data without further training.

    Loads the best agents, freezes their learned weights, and runs them
    through the specified data. Use this to test if a strategy generalizes.

    Examples:

        # Evaluate top 3 agents on default data
        vietlott-bingo18 eval

        # Evaluate top 5 agents on a different dataset
        vietlott-bingo18 eval --top-n 5 --data-path data/bingo18_2025.jsonl

        # Evaluate a specific agent
        vietlott-bingo18 eval --agent agent_012 --data-path data/bingo18_2025.jsonl
    """
    import json

    from machine_learning.bingo18.features import Bingo18FeatureEngineer

    state_dir = Path(agent_dir)
    data_path_obj = Path(data_path) if data_path else DEFAULT_DATA_PATH

    # Load data
    df = load_data(data_path_obj)

    # Load Stage 1 model
    model_path = state_dir / "stage1_model.joblib"
    model = Bingo18Model()
    if model_path.exists():
        logger.info(f"Loading Stage 1 model from {model_path}...")
        model.load(model_path)
    else:
        logger.info("No saved model found, training Stage 1 model...")
        model.train(df)

    # Select agent files to evaluate
    if agent:
        files = [state_dir / f"{agent}.json"]
        if not files[0].exists():
            click.echo(f"Agent file not found: {files[0]}")
            return
    else:
        all_files = sorted(state_dir.glob("*.json"))
        if not all_files:
            click.echo(f"No agent files found in {state_dir}")
            return
        # Sort by saved ROI to pick top-N
        scored = []
        for f in all_files:
            try:
                with f.open() as fp:
                    d = json.load(fp)
                s = d.get("state", {})
                budget = d.get("budget", 0)
                starting = s.get("starting_budget", budget)
                roi = (budget - starting) / starting * 100 if starting > 0 else 0.0
                scored.append((roi, f))
            except Exception:
                pass
        scored.sort(reverse=True)
        files = [f for _, f in scored[: top_n if top_n > 0 else len(scored)]]

    # Prepare data
    results_list = df["result"].tolist()
    totals = df["total"].tolist()
    large_smalls = df["large_small"].tolist()
    dates = df["date"].tolist() if "date" in df.columns else [f"draw_{i}" for i in range(len(df))]
    ids = df["id"].tolist() if "id" in df.columns else [f"{i:07d}" for i in range(len(df))]
    n_draws = len(results_list)

    from machine_learning.bingo18.agent import AdaptiveAgent

    click.echo(f"\n{'=' * 90}")
    click.echo(f"  EVAL: {len(files)} agent(s) on {data_path_obj.name}  ({n_draws} draws, frozen weights)")
    click.echo(f"{'=' * 90}")

    all_results = []
    for filepath in files:
        try:
            ag = AdaptiveAgent.load(filepath, model)
        except Exception as e:
            click.echo(f"  [WARN] Failed to load {filepath.name}: {e}")
            continue

        window = ag.genome.window
        feature_engineer = Bingo18FeatureEngineer(window=window)
        start_budget = ag._starting_budget

        # Reset all mutable state so eval results are clean (not polluted by training history)
        ag.budget = start_budget
        ag._wins = 0
        ag._losses = 0
        ag._total_bets = 0
        ag._current_streak = 0
        ag._profit_curve = [start_budget]
        ag._max_budget = start_budget
        ag._min_budget = start_budget
        ag._max_drawdown = 0
        ag._draws_since_adaptation = 0
        for stats in ag._bet_type_stats.values():
            stats.total_bets = 0
            stats.wins = 0
            stats.total_wagered = 0
            stats.total_payout = 0
            stats.recent_wins = []

        total_bets = 0

        for i in range(window, n_draws):
            recent_draws = results_list[i - window : i]
            recent_totals = totals[i - window : i]
            recent_ls = large_smalls[i - window : i]

            try:
                X = feature_engineer.build_features_for_predict(recent_draws, recent_totals, recent_ls)
            except (ValueError, IndexError):
                continue

            try:
                predictions = ag.model.predict_proba(X)
            except Exception:
                continue

            if not predictions:
                continue

            bets = ag.decide_bets(X, predictions)
            for bet_type, bet_value, bet_amount in bets:
                ag.record_result(
                    bet_type=bet_type,
                    bet_value=bet_value,
                    bet_amount=bet_amount,
                    actual_digits=results_list[i],
                    actual_total=totals[i],
                    date=dates[i],
                    draw_id=ids[i],
                )
                total_bets += 1
            # No increment_draw_counter, no maybe_adapt → weights frozen

        final_roi = (ag.budget - start_budget) / start_budget * 100 if start_budget > 0 else 0.0
        all_results.append(
            {
                "agent_id": ag.agent_id,
                "risk": ag.genome.risk_profile,
                "start_budget": start_budget,
                "final_budget": ag.budget,
                "roi": final_roi,
                "win_rate": ag.win_rate,
                "total_bets": total_bets,
                "bet_type_stats": ag._bet_type_stats,
            }
        )

    if not all_results:
        click.echo("  No results.")
        return

    all_results.sort(key=lambda r: r["roi"], reverse=True)

    click.echo(
        f"\n  {'Agent':<15} {'Risk':<13} {'Start Budget':>14} {'Final Budget':>14} {'ROI':>9} {'WinRate':>9} {'Bets':>7}"
    )
    click.echo(f"  {'-' * 85}")
    for r in all_results:
        roi_str = f"{r['roi']:>+8.1f}%"
        click.echo(
            f"  {r['agent_id']:<15} {r['risk']:<13} {r['start_budget']:>14,} "
            f"{r['final_budget']:>14,} {roi_str} {r['win_rate']:>8.1%} {r['total_bets']:>7,}"
        )

    click.echo(f"\n  Bet type breakdown (best agent: {all_results[0]['agent_id']}):")
    click.echo(f"  {'Bet Type':<20} {'ROI':>8} {'WinRate':>8} {'Bets':>8}")
    click.echo(f"  {'-' * 48}")
    best_stats = all_results[0]["bet_type_stats"]
    bt_rows = []
    for bt_name, stats in best_stats.items():
        if stats.total_bets > 0:
            bt_rows.append((bt_name, stats.roi, stats.win_rate, stats.total_bets))
    bt_rows.sort(key=lambda x: x[1], reverse=True)
    for bt_name, roi, wr, bets in bt_rows:
        click.echo(f"  {bt_name:<20} {roi:>+7.1f}% {wr:>7.1%} {bets:>8,}")

    click.echo(f"\n{'=' * 90}")


@cli.command()
@click.option("--n-agents", type=int, default=5, help="Number of survival agents")
@click.option("--rounds", type=int, default=3, help="Passes through data per agent")
@click.option("--budget", type=int, default=500_000, help="Starting budget per agent")
@click.option("--bet-size", type=int, default=10_000, help="VND per unit bet")
@click.option(
    "--bet-types",
    type=str,
    default=None,
    help="Comma-separated bet types to allow (default: all). "
    "Options: mot_so,hai_so_trung,ba_so_trung,cong_tong,lon_hoa_nho",
)
@click.option("--max-units-draw", type=int, default=5, help="Max units bet per draw")
@click.option("--max-units-option", type=int, default=3, help="Max units per bet option")
@click.option("--min-ratio", type=float, default=1.0, help="Min absence_ratio to trigger a bet")
@click.option(
    "--agent-dir",
    type=click.Path(),
    default="models/bingo18/survival",
    help="Directory to save/load agent state",
)
@click.option("--data-path", type=click.Path(exists=True), default=None, help="Path to bingo18.jsonl")
@click.option("--fresh", is_flag=True, help="Ignore saved state, start fresh")
def train_survival(
    n_agents: int,
    rounds: int,
    budget: int,
    bet_size: int,
    bet_types: str | None,
    max_units_draw: int,
    max_units_option: int,
    min_ratio: float,
    agent_dir: str,
    data_path: str | None,
    fresh: bool,
) -> None:
    """Train absence-based survival agents on historical Bingo18 data.

    Each agent tracks how long every bet option has been absent and bets on
    overdue outcomes.  More units are allocated to more-overdue options.
    No digit-prediction model is used — pure pattern-of-absence strategy.

    Examples:

        # Default: 5 agents, 3 rounds, all bet types
        vietlott-bingo18 train-survival

        # Restrict to specific bet types
        vietlott-bingo18 train-survival --bet-types mot_so,lon_hoa_nho

        # More aggressive: bet as soon as 0.8× expected gap
        vietlott-bingo18 train-survival --min-ratio 0.8 --max-units-draw 8

        # Continue from saved state
        vietlott-bingo18 train-survival --rounds 5
    """
    import time as _time

    from machine_learning.bingo18.survival_agent import SurvivalAgent

    state_dir = Path(agent_dir)
    data_path_obj = Path(data_path) if data_path else DEFAULT_DATA_PATH
    df = load_data(data_path_obj)

    # Convert to plain Python lists — pandas may return numpy arrays for list columns
    results_list = [list(r) for r in df["result"].tolist()]
    totals = df["total"].tolist()
    large_smalls = df["large_small"].tolist()
    dates = df["date"].tolist() if "date" in df.columns else [f"draw_{i}" for i in range(len(df))]
    ids = df["id"].tolist() if "id" in df.columns else [f"{i:07d}" for i in range(len(df))]
    n_draws = len(results_list)

    # Parse allowed bet types
    allowed_types: list[BetType] | None = None
    if bet_types:
        try:
            allowed_types = [BetType(bt.strip()) for bt in bet_types.split(",")]
        except ValueError as e:
            raise click.BadParameter(f"Unknown bet type: {e}") from e

    # Build diverse agent configs (vary min_absence_ratio and unit limits)
    def make_agent(idx: int) -> SurvivalAgent:
        ratio = min_ratio + idx * 0.1
        return SurvivalAgent(
            agent_id=f"survival_{idx:03d}",
            budget=budget,
            bet_size=bet_size,
            available_bet_types=allowed_types,
            max_units_per_draw=max_units_draw,
            max_units_per_option=max_units_option,
            min_absence_ratio=ratio,
        )

    # Load or create agents
    state_dir.mkdir(parents=True, exist_ok=True)
    agents: list[SurvivalAgent] = []
    for i in range(n_agents):
        filepath = state_dir / f"survival_{i:03d}.json"
        if not fresh and filepath.exists():
            try:
                agents.append(SurvivalAgent.load(filepath))
                continue
            except Exception as e:
                logger.warning(f"Failed to load {filepath.name}: {e}")
        agents.append(make_agent(i))

    click.echo(f"\n{'=' * 80}")
    click.echo(f"  SURVIVAL TRAINING  {n_agents} agents × {rounds} rounds  ({n_draws:,} draws)")
    click.echo(f"  bet_size={bet_size:,}  max_units/draw={max_units_draw}  min_ratio={min_ratio}")
    click.echo(f"{'=' * 80}")

    start_time = _time.perf_counter()

    for round_num in range(rounds):
        round_start = _time.perf_counter()
        bankruptcies = {a.agent_id: 0 for a in agents}

        for i in range(n_draws):
            for agent in agents:
                if not agent.is_alive:
                    bankruptcies[agent.agent_id] += 1
                    agent.reset_budget()
                agent.process_draw(results_list[i], totals[i], large_smalls[i], dates[i], ids[i])

        # Save all agents after each round
        for agent in agents:
            agent.save(state_dir)

        round_elapsed = _time.perf_counter() - round_start
        click.echo(f"\n  Round {round_num + 1}/{rounds}  ({round_elapsed:.1f}s)")
        click.echo(
            f"  {'Agent':<16} {'Budget':>12} {'SnapshotROI':>12} {'LifetimeROI':>12} "
            f"{'WinRate':>8} {'Bets':>7} {'Bankrupt':>9}"
        )
        click.echo(f"  {'-' * 80}")
        for agent in sorted(agents, key=lambda a: a.lifetime_roi, reverse=True):
            click.echo(
                f"  {agent.agent_id:<16} {agent.budget:>12,} "
                f"{agent.roi:>+11.1f}% {agent.lifetime_roi:>+11.1f}% "
                f"{agent.win_rate:>7.1%} {agent._total_bets:>7,} "
                f"{bankruptcies[agent.agent_id]:>8}"
            )

    total_elapsed = _time.perf_counter() - start_time
    click.echo(f"\n  Total time: {total_elapsed:.1f}s")
    click.echo(f"  Agent states saved to {state_dir}/")
    click.echo(f"{'=' * 80}")


@cli.command()
@click.option(
    "--agent-dir",
    type=click.Path(),
    default="models/bingo18/survival",
    help="Directory with saved survival agent state",
)
@click.option("--detail", is_flag=True, help="Show per-option breakdown for each agent")
@click.option("--top-options", type=int, default=5, help="Top N bet options to show per agent")
def survival_stats(agent_dir: str, detail: bool, top_options: int) -> None:
    """Show statistics for saved survival agents.

    Reads agent JSON files and prints a leaderboard sorted by lifetime ROI
    (most reliable metric, accumulated across all draws and bankruptcies).

    Examples:

        vietlott-bingo18 survival-stats
        vietlott-bingo18 survival-stats --detail --top-options 10
    """
    import json

    state_dir = Path(agent_dir)
    files = sorted(state_dir.glob("survival_*.json"))

    if not files:
        click.echo(f"No survival agent files found in {state_dir}")
        return

    rows = []
    for filepath in files:
        try:
            with filepath.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            click.echo(f"  [WARN] {filepath.name}: {e}")
            continue

        s = data.get("state", {})
        budget = data.get("budget", 0)
        starting = s.get("starting_budget", budget)
        snapshot_roi = (budget - starting) / starting * 100 if starting > 0 else 0.0

        opt_states = s.get("option_states", {})
        total_wagered = sum(v.get("total_wagered", 0) for v in opt_states.values())
        total_payout = sum(v.get("total_payout", 0) for v in opt_states.values())
        lifetime_roi = (total_payout - total_wagered) / total_wagered * 100 if total_wagered > 0 else 0.0

        # Top options by absence_ratio (most overdue right now)
        option_rows = []
        for key_str, od in opt_states.items():
            eg = od.get("expected_gap", 1)
            dsw = od.get("draws_since_win", 0)
            ratio = dsw / eg if eg > 0 else 0
            wagered = od.get("total_wagered", 0)
            payout = od.get("total_payout", 0)
            bets = od.get("total_bets", 0)
            opt_roi = (payout - wagered) / wagered * 100 if wagered > 0 else 0.0
            option_rows.append((key_str, ratio, dsw, eg, opt_roi, bets))

        option_rows.sort(key=lambda x: x[1], reverse=True)

        rows.append(
            {
                "agent_id": data.get("agent_id", "?"),
                "budget": budget,
                "starting": starting,
                "snapshot_roi": snapshot_roi,
                "lifetime_roi": lifetime_roi,
                "total_draws": s.get("total_draws", 0),
                "total_bets": s.get("total_bets", 0),
                "wins": s.get("wins", 0),
                "min_ratio": data.get("min_absence_ratio", 1.0),
                "max_units_draw": data.get("max_units_per_draw", 5),
                "option_rows": option_rows,
            }
        )

    if not rows:
        click.echo("No valid agent files found.")
        return

    rows.sort(key=lambda r: r["lifetime_roi"], reverse=True)

    click.echo(f"\n{'=' * 105}")
    click.echo(f"  SURVIVAL AGENT STATISTICS  ({len(rows)} agents from {state_dir})  [sorted by Lifetime ROI]")
    click.echo(f"{'=' * 105}")
    click.echo(
        f"  {'Agent':<16} {'Budget':>12} {'SnapshotROI':>12} {'LifetimeROI':>12} "
        f"{'WinRate':>8} {'Bets':>7} {'Draws':>7} {'MinRatio':>9}"
    )
    click.echo(f"  {'-' * 90}")

    for r in rows:
        wr = r["wins"] / r["total_bets"] if r["total_bets"] > 0 else 0.0
        click.echo(
            f"  {r['agent_id']:<16} {r['budget']:>12,} "
            f"{r['snapshot_roi']:>+11.1f}% {r['lifetime_roi']:>+11.1f}% "
            f"{wr:>7.1%} {r['total_bets']:>7,} {r['total_draws']:>7,} "
            f"{r['min_ratio']:>9.2f}"
        )

    click.echo(f"{'=' * 105}")

    # Aggregate by bet option
    agg: dict[str, dict] = {}
    for r in rows:
        for key_str, _, _, _, _, _ in r["option_rows"]:
            if key_str not in agg:
                agg[key_str] = {"wagered": 0, "payout": 0, "bets": 0}
    for filepath in files:
        try:
            with filepath.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for key_str, od in data.get("state", {}).get("option_states", {}).items():
                if key_str not in agg:
                    agg[key_str] = {"wagered": 0, "payout": 0, "bets": 0}
                agg[key_str]["wagered"] += od.get("total_wagered", 0)
                agg[key_str]["payout"] += od.get("total_payout", 0)
                agg[key_str]["bets"] += od.get("total_bets", 0)
        except Exception:
            pass

    agg_rows = [
        (k, (v["payout"] - v["wagered"]) / v["wagered"] * 100, v["bets"]) for k, v in agg.items() if v["wagered"] > 0
    ]
    agg_rows.sort(key=lambda x: x[1], reverse=True)

    click.echo(f"\n  BET OPTION PERFORMANCE (aggregate, top 10 and bottom 5)")
    click.echo(f"  {'Option':<30} {'ROI':>8} {'Total Bets':>12}")
    click.echo(f"  {'-' * 55}")
    for key_str, roi, bets in agg_rows[:10]:
        click.echo(f"  {key_str:<30} {roi:>+7.1f}% {bets:>12,}")
    if len(agg_rows) > 15:
        click.echo(f"  {'...':30}")
        for key_str, roi, bets in agg_rows[-5:]:
            click.echo(f"  {key_str:<30} {roi:>+7.1f}% {bets:>12,}")

    if detail:
        for r in rows:
            click.echo(
                f"\n  [{r['agent_id']}]  snapshot={r['snapshot_roi']:+.1f}%  "
                f"lifetime={r['lifetime_roi']:+.1f}%  bets={r['total_bets']:,}"
            )
            click.echo(f"  Most overdue right now (top {top_options}):")
            click.echo(f"    {'Option':<30} {'Absence':>8} {'Draws':>7} {'ExpGap':>8} {'ROI':>8} {'Bets':>6}")
            for key_str, ratio, dsw, eg, opt_roi, bets in r["option_rows"][:top_options]:
                click.echo(f"    {key_str:<30} {ratio:>8.2f}× {dsw:>7} {eg:>8.1f} {opt_roi:>+7.1f}% {bets:>6,}")


if __name__ == "__main__":
    cli()
