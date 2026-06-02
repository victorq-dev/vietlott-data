"""Visualization module for Bingo18 multi-agent racing system.

Provides charts for profit curves, bet decisions, win/loss streaks,
adaptation events, leaderboards, and multi-agent comparisons.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.figure import Figure

from machine_learning.bingo18.simulator import BetRecord

# Consistent color palette
COLOR_WIN = "#2ecc71"  # green
COLOR_LOSS = "#e74c3c"  # red
COLOR_NEUTRAL = "#3498db"  # blue
COLOR_START = "#95a5a6"  # gray
COLORS_BET_TYPE = {
    "mot_so": "#3498db",
    "hai_so_trung": "#2ecc71",
    "ba_so_trung": "#e74c3c",
    "cong_tong": "#f39c12",
    "lon_hoa_nho": "#9b59b6",
}
DEFAULT_DPI = 150
FIG_SIZE_DEFAULT = (12, 8)
FIG_SIZE_LARGE = (16, 12)


def _apply_style() -> None:
    """Apply consistent chart styling."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
        }
    )


def _empty_figure(message: str = "No data available") -> Figure:
    """Return an empty figure with a message."""
    fig, ax = plt.subplots(figsize=FIG_SIZE_DEFAULT)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=14, color="#7f8c8d", transform=ax.transAxes)
    ax.set_axis_off()
    return fig


def _roi_color(roi: float) -> str:
    """Return color based on ROI value."""
    return COLOR_WIN if roi >= 0 else COLOR_LOSS


def plot_profit_curves(results: dict[str, list[int]], title: str = "Agent Profit Curves") -> Figure:
    """Plot multi-line profit curves for each agent.

    Parameters
    ----------
    results : dict[str, list[int]]
        Mapping of agent_id -> list of budget values over draws.
    title : str
        Chart title.

    Returns
    -------
    Figure
    """
    _apply_style()

    if not results:
        return _empty_figure("No agent results to plot")

    cmap = plt.cm.get_cmap("tab10", max(len(results), 10))

    fig, ax = plt.subplots(figsize=FIG_SIZE_DEFAULT)

    # Determine starting budget from first value of each agent
    starting_budgets = [vals[0] for vals in results.values() if vals]
    if not starting_budgets:
        return _empty_figure("No profit curve data")

    starting_budget = starting_budgets[0]

    for i, (agent_id, budgets) in enumerate(results.items()):
        if not budgets:
            continue
        roi = (budgets[-1] - budgets[0]) / budgets[0] * 100 if budgets[0] > 0 else 0
        color = cmap(i)
        ax.plot(range(len(budgets)), budgets, label=agent_id, color=color, linewidth=1.5, alpha=0.85)

        # Highlight max drawdown region
        max_dd_start = 0
        max_dd_end = 0
        max_dd_val = 0
        peak = budgets[0]
        peak_idx = 0
        for i, val in enumerate(budgets):
            if val > peak:
                peak = val
                peak_idx = i
            dd = peak - val
            if dd > max_dd_val:
                max_dd_val = dd
                max_dd_start = peak_idx
                max_dd_end = i

        if max_dd_val > 0:
            ax.axvspan(max_dd_start, max_dd_end, alpha=0.08, color=COLOR_LOSS)

    # Horizontal line at starting budget
    ax.axhline(
        y=starting_budget, color=COLOR_START, linestyle="--", linewidth=1, label=f"Start: {starting_budget:,} VND"
    )

    ax.set_xlabel("Draw Index")
    ax.set_ylabel("Budget (VND)")
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    return fig


def plot_bet_decisions(bet_history: list[BetRecord], title: str = "Bet Decisions Timeline") -> Figure:
    """Plot scatter of bet decisions over time.

    Parameters
    ----------
    bet_history : list[BetRecord]
        List of bet records.
    title : str
        Chart title.

    Returns
    -------
    Figure
    """
    _apply_style()

    if not bet_history:
        return _empty_figure("No bet history to plot")

    fig, ax = plt.subplots(figsize=FIG_SIZE_DEFAULT)

    bet_types = sorted(set(b.bet_type for b in bet_history))
    type_to_y = {bt: i for i, bt in enumerate(bet_types)}

    wins_x, wins_y, wins_s = [], [], []
    losses_x, losses_y, losses_s = [], [], []

    for i, bet in enumerate(bet_history):
        y = type_to_y[bet.bet_type]
        size = max(20, bet.bet_amount / 500)
        if bet.payout > 0:
            wins_x.append(i)
            wins_y.append(y)
            wins_s.append(size)
        else:
            losses_x.append(i)
            losses_y.append(y)
            losses_s.append(size)

    if wins_x:
        ax.scatter(wins_x, wins_y, s=wins_s, c=COLOR_WIN, alpha=0.7, label="Win", edgecolors="white", linewidth=0.5)
    if losses_x:
        ax.scatter(
            losses_x, losses_y, s=losses_s, c=COLOR_LOSS, alpha=0.7, label="Loss", edgecolors="white", linewidth=0.5
        )

    ax.set_yticks(range(len(bet_types)))
    ax.set_yticklabels(bet_types)
    ax.set_xlabel("Bet Index")
    ax.set_ylabel("Bet Type")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_win_loss_streaks(bet_history: list[BetRecord], title: str = "Win/Loss Streaks") -> Figure:
    """Plot bar chart of win/loss streak lengths.

    Parameters
    ----------
    bet_history : list[BetRecord]
        List of bet records.
    title : str
        Chart title.

    Returns
    -------
    Figure
    """
    _apply_style()

    if not bet_history:
        return _empty_figure("No bet history to plot")

    # Calculate streaks
    streaks: list[int] = []
    current_streak = 0
    for bet in bet_history:
        if bet.payout > 0:
            if current_streak >= 0:
                current_streak += 1
            else:
                streaks.append(current_streak)
                current_streak = 1
        else:
            if current_streak <= 0:
                current_streak -= 1
            else:
                streaks.append(current_streak)
                current_streak = -1
    if current_streak != 0:
        streaks.append(current_streak)

    if not streaks:
        return _empty_figure("No streaks found")

    fig, ax = plt.subplots(figsize=FIG_SIZE_DEFAULT)

    colors = [COLOR_WIN if s > 0 else COLOR_LOSS for s in streaks]
    ax.bar(range(len(streaks)), streaks, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Streak Number")
    ax.set_ylabel("Streak Length (wins+/losses-)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_adaptation_events(adaptation_log: list[dict], title: str = "Adaptation Events") -> Figure:
    """Plot timeline of adaptation events.

    Parameters
    ----------
    adaptation_log : list[dict]
        Each dict should have 'draw_index', 'generation', and optionally 'description'.
    title : str
        Chart title.

    Returns
    -------
    Figure
    """
    _apply_style()

    if not adaptation_log:
        return _empty_figure("No adaptation events to plot")

    fig, ax = plt.subplots(figsize=FIG_SIZE_DEFAULT)

    draw_indices = [e.get("draw_index", 0) for e in adaptation_log]
    generations = [e.get("generation", 0) for e in adaptation_log]
    descriptions = [e.get("description", "") for e in adaptation_log]

    ax.scatter(draw_indices, generations, s=100, c=COLOR_NEUTRAL, alpha=0.8, edgecolors="white", linewidth=1, zorder=5)
    ax.plot(draw_indices, generations, color=COLOR_NEUTRAL, alpha=0.3, linewidth=1)

    # Annotate key events
    for i, (x, y, desc) in enumerate(zip(draw_indices, generations, descriptions)):
        if desc and i % max(1, len(adaptation_log) // 10) == 0:
            ax.annotate(
                desc,
                (x, y),
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                fontsize=8,
                alpha=0.7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    ax.set_xlabel("Draw Index")
    ax.set_ylabel("Generation")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_leaderboard(agent_results: list[dict], title: str = "Agent Leaderboard") -> Figure:
    """Plot horizontal bar chart of agents sorted by ROI.

    Parameters
    ----------
    agent_results : list[dict]
        Each dict should have 'agent_id', 'roi', 'final_budget', 'win_rate'.
    title : str
        Chart title.

    Returns
    -------
    Figure
    """
    _apply_style()

    if not agent_results:
        return _empty_figure("No agent results to plot")

    # Sort by ROI descending
    sorted_results = sorted(agent_results, key=lambda r: r.get("roi", 0), reverse=True)

    agent_ids = [r.get("agent_id", f"Agent {i}") for i, r in enumerate(sorted_results)]
    rois = [r.get("roi", 0) for r in sorted_results]
    final_budgets = [r.get("final_budget", 0) for r in sorted_results]
    win_rates = [r.get("win_rate", 0) for r in sorted_results]

    # Color gradient: green for best, red for worst
    n = len(sorted_results)
    if n > 1:
        colors = []
        for i in range(n):
            ratio = i / (n - 1)
            r = int(46 + ratio * (231 - 46))
            g = int(204 + ratio * (76 - 204))
            b = int(113 + ratio * (60 - 113))
            colors.append(f"#{r:02x}{g:02x}{b:02x}")
    else:
        colors = [COLOR_WIN]

    fig, ax = plt.subplots(figsize=(14, max(6, n * 0.5)))

    y_pos = range(n)
    bars = ax.barh(y_pos, rois, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)

    # Add text labels
    for i, (roi, budget, wr) in enumerate(zip(rois, final_budgets, win_rates)):
        label = f"ROI: {roi:+.1f}%  |  Budget: {budget:,}  |  WinRate: {wr:.1%}"
        x_pos = roi + (max(rois) - min(rois)) * 0.01 if roi >= 0 else roi - (max(rois) - min(rois)) * 0.01
        ha = "left" if roi >= 0 else "right"
        ax.text(x_pos, i, label, va="center", ha=ha, fontsize=9, alpha=0.8)

    # Highlight top 3
    for i in range(min(3, n)):
        bars[i].set_edgecolor(COLOR_WIN)
        bars[i].set_linewidth(2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(agent_ids)
    ax.set_xlabel("ROI (%)")
    ax.set_title(title)
    ax.axvline(x=0, color="black", linewidth=0.5)
    fig.tight_layout()
    return fig


def plot_bet_type_distribution(bet_history: list[BetRecord], title: str = "Bet Type Distribution") -> Figure:
    """Plot distribution of bet types with win rates.

    Parameters
    ----------
    bet_history : list[BetRecord]
        List of bet records.
    title : str
        Chart title.

    Returns
    -------
    Figure
    """
    _apply_style()

    if not bet_history:
        return _empty_figure("No bet history to plot")

    # Aggregate by bet type
    type_stats: dict[str, dict[str, int]] = {}
    for bet in bet_history:
        bt = bet.bet_type
        if bt not in type_stats:
            type_stats[bt] = {"total": 0, "wins": 0}
        type_stats[bt]["total"] += 1
        if bet.payout > 0:
            type_stats[bt]["wins"] += 1

    bet_types = sorted(type_stats.keys())
    totals = [type_stats[bt]["total"] for bt in bet_types]
    win_rates = [
        type_stats[bt]["wins"] / type_stats[bt]["total"] if type_stats[bt]["total"] > 0 else 0 for bt in bet_types
    ]
    colors = [COLORS_BET_TYPE.get(bt, COLOR_NEUTRAL) for bt in bet_types]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE_LARGE, gridspec_kw={"width_ratios": [2, 1]})

    # Bar chart with counts
    bars = ax1.bar(range(len(bet_types)), totals, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    for i, (bar, wr) in enumerate(zip(bars, win_rates)):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(totals) * 0.01,
            f"WR: {wr:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
            alpha=0.8,
        )

    ax1.set_xticks(range(len(bet_types)))
    ax1.set_xticklabels(bet_types, rotation=45, ha="right")
    ax1.set_ylabel("Number of Bets")
    ax1.set_title(f"{title} - Counts")

    # Pie chart
    ax2.pie(totals, labels=bet_types, colors=colors, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9})
    ax2.set_title(f"{title} - Share")

    fig.tight_layout()
    return fig


def plot_multi_agent_comparison(agent_results: list[dict], title: str = "Multi-Agent Comparison") -> Figure:
    """Plot 2x2 subplot grid comparing agents across key metrics.

    Parameters
    ----------
    agent_results : list[dict]
        Each dict should have 'agent_id', 'roi', 'win_rate', 'max_drawdown', 'total_bets'.
    title : str
        Chart title.

    Returns
    -------
    Figure
    """
    _apply_style()

    if not agent_results:
        return _empty_figure("No agent results to compare")

    agent_ids = [r.get("agent_id", f"Agent {i}") for i, r in enumerate(agent_results)]
    rois = [r.get("roi", 0) for r in agent_results]
    win_rates = [r.get("win_rate", 0) * 100 for r in agent_results]
    drawdowns = [r.get("max_drawdown", 0) for r in agent_results]
    total_bets = [r.get("total_bets", 0) for r in agent_results]

    n = len(agent_results)
    x = np.arange(n)
    width = max(0.3, 0.8 / n)

    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE_LARGE)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # ROI
    ax = axes[0, 0]
    colors = [_roi_color(r) for r in rois]
    ax.bar(x, rois, width, color=colors, alpha=0.85, edgecolor="white")
    ax.set_title("ROI (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(agent_ids, rotation=45, ha="right", fontsize=8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    for i, v in enumerate(rois):
        ax.text(i, v + (max(rois) - min(rois)) * 0.02, f"{v:+.1f}%", ha="center", va="bottom", fontsize=8)

    # Win Rate
    ax = axes[0, 1]
    ax.bar(x, win_rates, width, color=COLOR_WIN, alpha=0.85, edgecolor="white")
    ax.set_title("Win Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(agent_ids, rotation=45, ha="right", fontsize=8)
    for i, v in enumerate(win_rates):
        ax.text(i, v + max(win_rates) * 0.02, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    # Max Drawdown
    ax = axes[1, 0]
    ax.bar(x, drawdowns, width, color=COLOR_LOSS, alpha=0.85, edgecolor="white")
    ax.set_title("Max Drawdown (VND)")
    ax.set_xticks(x)
    ax.set_xticklabels(agent_ids, rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:,.0f}"))
    for i, v in enumerate(drawdowns):
        ax.text(i, v + max(drawdowns) * 0.02, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)

    # Total Bets
    ax = axes[1, 1]
    ax.bar(x, total_bets, width, color=COLOR_NEUTRAL, alpha=0.85, edgecolor="white")
    ax.set_title("Total Bets")
    ax.set_xticks(x)
    ax.set_xticklabels(agent_ids, rotation=45, ha="right", fontsize=8)
    for i, v in enumerate(total_bets):
        ax.text(i, v + max(total_bets) * 0.02, f"{v:,}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def generate_race_report(
    agent_results: list[dict],
    output_dir: Path,
    starting_budget: int,
    profit_curves: dict[str, list[int]] | None = None,
    bet_history: list[BetRecord] | None = None,
    adaptation_log: list[dict] | None = None,
) -> None:
    """Generate full race report with all charts and summary markdown.

    Parameters
    ----------
    agent_results : list[dict]
        Each dict should have 'agent_id', 'roi', 'final_budget', 'win_rate',
        'max_drawdown', 'total_bets', and optionally 'bet_type', 'algorithm'.
    output_dir : Path
        Directory to save charts and report.
    starting_budget : int
        Starting budget for all agents.
    profit_curves : dict[str, list[int]] | None
        Optional profit curves for each agent.
    bet_history : list[BetRecord] | None
        Optional combined bet history for distribution chart.
    adaptation_log : list[dict] | None
        Optional adaptation events log.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    logger.info(f"Generating race report to {output_dir}")

    # Save individual charts
    charts_generated = []

    # 1. Profit curves
    if profit_curves:
        fig = plot_profit_curves(profit_curves, title="Agent Profit Curves")
        path = output_dir / "profit_curves.png"
        fig.savefig(path, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)
        charts_generated.append("profit_curves.png")
        logger.info(f"Saved {path}")

    # 2. Leaderboard
    fig = plot_leaderboard(agent_results, title="Agent Leaderboard")
    path = output_dir / "leaderboard.png"
    fig.savefig(path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    charts_generated.append("leaderboard.png")

    # 3. Multi-agent comparison
    if len(agent_results) > 1:
        fig = plot_multi_agent_comparison(agent_results, title="Multi-Agent Comparison")
        path = output_dir / "comparison.png"
        fig.savefig(path, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)
        charts_generated.append("comparison.png")

    # 4. Bet type distribution
    if bet_history:
        fig = plot_bet_type_distribution(bet_history, title="Bet Type Distribution")
        path = output_dir / "bet_distribution.png"
        fig.savefig(path, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)
        charts_generated.append("bet_distribution.png")

    # 5. Bet decisions
    if bet_history:
        fig = plot_bet_decisions(bet_history, title="Bet Decisions Timeline")
        path = output_dir / "bet_decisions.png"
        fig.savefig(path, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)
        charts_generated.append("bet_decisions.png")

    # 6. Win/loss streaks
    if bet_history:
        fig = plot_win_loss_streaks(bet_history, title="Win/Loss Streaks")
        path = output_dir / "streaks.png"
        fig.savefig(path, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)
        charts_generated.append("streaks.png")

    # 7. Adaptation events
    if adaptation_log:
        fig = plot_adaptation_events(adaptation_log, title="Adaptation Events")
        path = output_dir / "adaptations.png"
        fig.savefig(path, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)
        charts_generated.append("adaptations.png")

    # Generate summary markdown
    summary_lines = [
        "# Bingo18 Multi-Agent Race Report",
        "",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        f"- Starting budget: {starting_budget:,} VND",
        f"- Number of agents: {len(agent_results)}",
        "",
    ]

    # Leaderboard table
    sorted_results = sorted(agent_results, key=lambda r: r.get("roi", 0), reverse=True)
    summary_lines.append("## Leaderboard")
    summary_lines.append("")
    summary_lines.append("| Rank | Agent | ROI | Final Budget | Win Rate | Max Drawdown | Total Bets |")
    summary_lines.append("|------|-------|-----|--------------|----------|--------------|------------|")

    for i, r in enumerate(sorted_results, 1):
        agent_id = r.get("agent_id", f"Agent {i}")
        roi = r.get("roi", 0)
        final = r.get("final_budget", 0)
        wr = r.get("win_rate", 0)
        dd = r.get("max_drawdown", 0)
        bets = r.get("total_bets", 0)
        summary_lines.append(f"| {i} | {agent_id} | {roi:+.2f}% | {final:,} | {wr:.1%} | {dd:,} | {bets:,} |")

    summary_lines.append("")

    # Key findings
    if sorted_results:
        best = sorted_results[0]
        worst = sorted_results[-1]
        profitable = sum(1 for r in sorted_results if r.get("roi", 0) > 0)

        summary_lines.extend(
            [
                "## Key Findings",
                "",
                f"- Best agent: **{best.get('agent_id', 'Unknown')}** with ROI {best.get('roi', 0):+.2f}%",
                f"- Worst agent: **{worst.get('agent_id', 'Unknown')}** with ROI {worst.get('roi', 0):+.2f}%",
                f"- Profitable agents: {profitable}/{len(sorted_results)}",
                f"- Average ROI: {np.mean([r.get('roi', 0) for r in sorted_results]):+.2f}%",
                "",
            ]
        )

    # Charts list
    summary_lines.extend(
        [
            "## Charts",
            "",
        ]
    )
    for chart in charts_generated:
        summary_lines.append(f"- ![{chart.replace('.png', '')}]({chart})")

    summary_lines.extend(
        [
            "",
            "---",
            "",
            "*Disclaimer: Lottery outcomes are random. These results are for educational/research purposes only.*",
        ]
    )

    report_path = output_dir / "race_report.md"
    report_path.write_text("\n".join(summary_lines), encoding="utf-8")
    logger.info(f"Race report saved to {report_path}")


def export_agent_decisions_csv(
    agent_results: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    """Export per-agent bet decisions to CSV files.

    Parameters
    ----------
    agent_results : list[dict]
        Each dict needs 'agent_id' and 'bet_history' (list[BetRecord]).
    output_dir : Path
        Directory to save CSV files.

    Returns
    -------
    list[Path]
        Paths of saved CSV files.
    """
    import csv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for ar in agent_results:
        agent_id = ar["agent_id"]
        bet_history = ar.get("bet_history", [])
        if not bet_history:
            continue

        csv_path = output_dir / f"{agent_id}_decisions.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "draw_id",
                    "date",
                    "bet_type",
                    "bet_value",
                    "actual_digits",
                    "actual_total",
                    "matches",
                    "bet_amount",
                    "payout",
                    "profit",
                    "budget_after",
                ]
            )
            for bet in bet_history:
                writer.writerow(
                    [
                        bet.draw_id,
                        bet.date,
                        bet.bet_type,
                        bet.bet_value,
                        bet.actual_digits,
                        bet.actual_total,
                        bet.matches,
                        bet.bet_amount,
                        bet.payout,
                        bet.payout - bet.bet_amount,
                        bet.budget_after,
                    ]
                )
        saved.append(csv_path)
        logger.info(f"Saved {csv_path} ({len(bet_history)} decisions)")

    return saved
