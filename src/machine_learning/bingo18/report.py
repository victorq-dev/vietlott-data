"""Render simulation results as markdown report."""

from datetime import datetime
from pathlib import Path

from loguru import logger
from tabulate import tabulate

from machine_learning.bingo18.simulator import SimulationResult


def render_summary(result: SimulationResult) -> str:
    """Render simulation summary as markdown."""
    lines = [
        "# Bingo18 Auto-Play Simulation Report",
        "",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        tabulate(
            [
                ["Starting Budget", f"{result.starting_budget:,} VND"],
                ["Final Budget", f"{result.final_budget:,} VND"],
                ["Profit/Loss", f"{result.profit:,} VND"],
                ["ROI", f"{result.roi:.2f}%"],
                ["Bet Size", f"{result.bet_size:,} VND"],
                ["Bet Type", result.bet_type],
                ["Total Bets", f"{result.total_bets:,}"],
                ["Wins", f"{result.wins:,}"],
                ["Losses", f"{result.losses:,}"],
                ["Win Rate", f"{result.win_rate:.2%}"],
                ["Max Drawdown", f"{result.max_drawdown:,} VND"],
            ],
            tablefmt="pipe",
        ),
    ]
    return "\n".join(lines)


def render_bet_history(result: SimulationResult, last_n: int = 20) -> str:
    """Render recent bet history as markdown table."""
    if not result.bet_history:
        return ""

    lines = [
        "",
        "## Recent Bets (last {})".format(min(last_n, len(result.bet_history))),
        "",
    ]

    headers = ["Date", "Bet Type", "Bet Value", "Actual", "Total", "Payout", "Budget"]
    rows = []
    for bet in result.bet_history[-last_n:]:
        rows.append(
            [
                bet.date,
                bet.bet_type,
                str(bet.bet_value),
                str(bet.actual_digits),
                bet.actual_total,
                f"{bet.payout:,}",
                f"{bet.budget_after:,}",
            ]
        )

    lines.append(tabulate(rows, headers=headers, tablefmt="pipe"))
    return "\n".join(lines)


def render_match_distribution(result: SimulationResult) -> str:
    """Render match count distribution."""
    if not result.bet_history:
        return ""

    # Group by payout amount for more meaningful distribution
    payout_dist: dict[int, int] = {}
    for bet in result.bet_history:
        payout_dist[bet.payout] = payout_dist.get(bet.payout, 0) + 1

    lines = [
        "",
        "## Payout Distribution",
        "",
        tabulate(
            [[f"{k:,} VND", f"{v:,}", f"{v / result.total_bets:.2%}"] for k, v in sorted(payout_dist.items())],
            headers=["Payout", "Count", "Rate"],
            tablefmt="pipe",
        ),
    ]
    return "\n".join(lines)


def render_report(result: SimulationResult) -> str:
    """Render full simulation report."""
    parts = [
        render_summary(result),
        render_match_distribution(result),
        render_bet_history(result),
        "",
        "---",
        "",
        "*Disclaimer: This simulation is for educational purposes only. "
        "Lottery outcomes are random and cannot be predicted reliably.*",
    ]
    return "\n".join(parts)


def save_report(result: SimulationResult, path: Path) -> None:
    """Save report to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = render_report(result)
    path.write_text(report, encoding="utf-8")
    logger.info(f"Report saved to {path}")
