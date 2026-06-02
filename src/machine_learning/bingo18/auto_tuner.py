"""Auto-tuner for Bingo18: try multiple algorithms and strategies to find the best."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from tabulate import tabulate

from machine_learning.bingo18.model import Bingo18Model
from machine_learning.bingo18.simulator import Bingo18Simulator

# Default search space
DEFAULT_ALGORITHMS = ["gradient_boosting", "random_forest", "extra_trees", "logistic_regression"]
DEFAULT_WINDOWS = [10, 30, 50]
DEFAULT_N_ESTIMATORS = [50, 100]
DEFAULT_MAX_DEPTH = [3, 5]
DEFAULT_BET_TYPES = [
    "mot_so",
    "hai_so_trung",
    "ba_so_trung",
    "cong_tong",
    "lon_hoa_nho",

]
DEFAULT_STRATEGIES = ["top_n", "threshold"]
DEFAULT_THRESHOLDS = [0.12, 0.15]
DEFAULT_BUDGET_LEVELS = [1_000_000, 5_000_000, 10_000_000, 50_000_000]


@dataclass
class TunerResult:
    """Result of a single tuning run."""

    algorithm: str
    window: int
    n_estimators: int
    max_depth: int
    bet_type: str
    strategy: str
    threshold: float
    budget: int
    final_budget: int
    roi: float
    bets_survived: int
    win_rate: float
    max_drawdown: int
    total_bets: int
    model_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TunerSummary:
    """Summary of auto-tuning run."""

    total_combinations: int
    total_simulations: int
    budget_levels: list[int]
    best_by_budget: dict[int, TunerResult]  # budget -> best result
    top_results: list[TunerResult]
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_combinations": self.total_combinations,
            "total_simulations": self.total_simulations,
            "budget_levels": self.budget_levels,
            "best_by_budget": {str(k): v.to_dict() for k, v in self.best_by_budget.items()},
            "top_results": [r.to_dict() for r in self.top_results],
            "timestamp": self.timestamp,
        }


class Bingo18AutoTuner:
    """Auto-tuner that tries multiple algorithms and strategies.

    For each combination of (algorithm, params, bet_type, strategy),
    runs simulation with multiple budget levels and finds the best.
    """

    def __init__(
        self,
        bet_size: int = 10_000,
        algorithms: list[str] | None = None,
        windows: list[int] | None = None,
        n_estimators: list[int] | None = None,
        max_depths: list[int] | None = None,
        bet_types: list[str] | None = None,
        strategies: list[str] | None = None,
        thresholds: list[float] | None = None,
        budget_levels: list[int] | None = None,
    ):
        self.bet_size = bet_size
        self.algorithms = algorithms or DEFAULT_ALGORITHMS
        self.windows = windows or DEFAULT_WINDOWS
        self.n_estimators = n_estimators or DEFAULT_N_ESTIMATORS
        self.max_depths = max_depths or DEFAULT_MAX_DEPTH
        self.bet_types = bet_types or DEFAULT_BET_TYPES
        self.strategies = strategies or DEFAULT_STRATEGIES
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.budget_levels = budget_levels or DEFAULT_BUDGET_LEVELS

    def _get_search_space(self) -> list[dict[str, Any]]:
        """Generate all combinations of parameters."""
        combinations = []

        for algo, window, n_est, depth in product(self.algorithms, self.windows, self.n_estimators, self.max_depths):
            # LogisticRegression doesn't use n_estimators/max_depth
            if algo == "logistic_regression":
                model_params = {"window": window, "algorithm": algo}
            else:
                model_params = {"window": window, "algorithm": algo, "n_estimators": n_est, "max_depth": depth}

            for bet_type in self.bet_types:
                for strategy in self.strategies:
                    if strategy == "threshold":
                        for threshold in self.thresholds:
                            combinations.append(
                                {
                                    "model_params": model_params,
                                    "bet_type": bet_type,
                                    "strategy": strategy,
                                    "threshold": threshold,
                                }
                            )
                    else:
                        combinations.append(
                            {
                                "model_params": model_params,
                                "bet_type": bet_type,
                                "strategy": strategy,
                                "threshold": 0.12,
                            }
                        )

        return combinations

    def run(self, df: pd.DataFrame, top_k: int = 10) -> TunerSummary:
        """Run auto-tuning across all combinations and budget levels.

        Parameters
        ----------
        df : pd.DataFrame
            Bingo18 data.
        top_k : int
            Number of top results to return.

        Returns
        -------
        TunerSummary with results.
        """
        search_space = self._get_search_space()
        total_sims = len(search_space) * len(self.budget_levels)

        logger.info(
            f"Auto-tuner: {len(search_space)} combinations × {len(self.budget_levels)} budgets "
            f"= {total_sims} simulations"
        )

        all_results: list[TunerResult] = []
        completed = 0

        for combo in search_space:
            model_params = combo["model_params"]
            bet_type = combo["bet_type"]
            strategy = combo["strategy"]
            threshold = combo["threshold"]

            # Train model once per model_params combination
            try:
                model = Bingo18Model(**model_params)
                model.train(df)
            except Exception as e:
                logger.warning(f"Failed to train {model_params}: {e}")
                completed += len(self.budget_levels)
                continue

            for budget in self.budget_levels:
                try:
                    sim = Bingo18Simulator(
                        model=model,
                        budget=budget,
                        bet_size=self.bet_size,
                        bet_type=bet_type,
                        strategy=strategy,
                        threshold=threshold,
                    )
                    result = sim.run(df)

                    tuner_result = TunerResult(
                        algorithm=model_params.get("algorithm", "gradient_boosting"),
                        window=model_params.get("window", 30),
                        n_estimators=model_params.get("n_estimators", 0),
                        max_depth=model_params.get("max_depth", 0),
                        bet_type=bet_type,
                        strategy=strategy,
                        threshold=threshold,
                        budget=budget,
                        final_budget=result.final_budget,
                        roi=result.roi,
                        bets_survived=result.total_bets,
                        win_rate=result.win_rate,
                        max_drawdown=result.max_drawdown,
                        total_bets=result.total_bets,
                    )
                    all_results.append(tuner_result)
                except Exception as e:
                    logger.warning(f"Simulation failed: {e}")

                completed += 1
                if completed % 50 == 0:
                    logger.info(f"Progress: {completed}/{total_sims}")

        # Sort by: 1) final_budget desc, 2) bets_survived desc
        all_results.sort(key=lambda r: (r.final_budget, r.bets_survived), reverse=True)

        # Find best per budget level
        best_by_budget: dict[int, TunerResult] = {}
        for budget in self.budget_levels:
            budget_results = [r for r in all_results if r.budget == budget]
            if budget_results:
                best_by_budget[budget] = budget_results[0]

        summary = TunerSummary(
            total_combinations=len(search_space),
            total_simulations=total_sims,
            budget_levels=self.budget_levels,
            best_by_budget=best_by_budget,
            top_results=all_results[:top_k],
        )

        logger.info("Auto-tuning complete. Best results:")
        for budget, best in best_by_budget.items():
            logger.info(
                f"  Budget {budget:,}: {best.bet_type}/{best.strategy} -> {best.final_budget:,} VND (ROI: {best.roi:.1f}%)"
            )

        return summary

    def run_combined(
        self,
        df: pd.DataFrame,
        combined_configs: list[dict[str, Any]],
        top_k: int = 10,
    ) -> TunerSummary:
        """Run auto-tuning with combined betting modes.

        Parameters
        ----------
        df : pd.DataFrame
            Bingo18 data.
        combined_configs : list[dict]
            Each dict has: bet_types (list[str]), mode (str), confidence (float, optional)
        top_k : int
            Number of top results to return.
        """
        # NOTE: iteration pattern mirrors run() above; consider extracting a shared helper
        # if more modes are added in the future.
        search_space = self._get_search_space()
        # Add combined configs to search space
        for config in combined_configs:
            for algo, window, n_est, depth in product(
                self.algorithms, self.windows, self.n_estimators, self.max_depths
            ):
                if algo == "logistic_regression":
                    model_params = {"window": window, "algorithm": algo}
                else:
                    model_params = {"window": window, "algorithm": algo, "n_estimators": n_est, "max_depth": depth}
                search_space.append(
                    {
                        "model_params": model_params,
                        "bet_type": "+".join(config["bet_types"]),
                        "strategy": config.get("mode", "combine"),
                        "threshold": config.get("confidence", 0.0),
                        "combined": config,
                    }
                )

        total_sims = len(search_space) * len(self.budget_levels)
        logger.info(
            f"Combined auto-tuner: {len(search_space)} combinations × {len(self.budget_levels)} budgets = {total_sims}"
        )

        all_results: list[TunerResult] = []
        completed = 0

        for combo in search_space:
            model_params = combo["model_params"]
            bet_type = combo["bet_type"]
            strategy = combo["strategy"]
            threshold = combo["threshold"]
            combined = combo.get("combined")

            try:
                model = Bingo18Model(**model_params)
                model.train(df)
            except Exception as e:
                logger.warning(f"Failed to train {model_params}: {e}")
                completed += len(self.budget_levels)
                continue

            for budget in self.budget_levels:
                try:
                    sim = Bingo18Simulator(
                        model=model,
                        budget=budget,
                        bet_size=self.bet_size,
                    )

                    if combined:
                        result = sim.run_combined(
                            df,
                            bet_types=combined["bet_types"],
                            mode=combined.get("mode", "combine"),
                            confidence_threshold=combined.get("confidence", 0.0),
                        )
                    else:
                        result = sim.run(df)

                    tuner_result = TunerResult(
                        algorithm=model_params.get("algorithm", "gradient_boosting"),
                        window=model_params.get("window", 30),
                        n_estimators=model_params.get("n_estimators", 0),
                        max_depth=model_params.get("max_depth", 0),
                        bet_type=bet_type,
                        strategy=strategy,
                        threshold=threshold,
                        budget=budget,
                        final_budget=result.final_budget,
                        roi=result.roi,
                        bets_survived=result.total_bets,
                        win_rate=result.win_rate,
                        max_drawdown=result.max_drawdown,
                        total_bets=result.total_bets,
                    )
                    all_results.append(tuner_result)
                except Exception as e:
                    logger.warning(f"Simulation failed: {e}")

                completed += 1
                if completed % 50 == 0:
                    logger.info(f"Progress: {completed}/{total_sims}")

        all_results.sort(key=lambda r: (r.final_budget, r.bets_survived), reverse=True)

        best_by_budget: dict[int, TunerResult] = {}
        for budget in self.budget_levels:
            budget_results = [r for r in all_results if r.budget == budget]
            if budget_results:
                best_by_budget[budget] = budget_results[0]

        summary = TunerSummary(
            total_combinations=len(search_space),
            total_simulations=total_sims,
            budget_levels=self.budget_levels,
            best_by_budget=best_by_budget,
            top_results=all_results[:top_k],
        )

        logger.info("Combined auto-tuning complete. Best results:")
        for budget, best in best_by_budget.items():
            logger.info(
                f"  Budget {budget:,}: {best.algorithm} + {best.bet_type} + {best.strategy} "
                f"→ final={best.final_budget:,}, ROI={best.roi:.2f}%, bets={best.bets_survived}"
            )

        return summary

    def run_and_save(self, df: pd.DataFrame, save_dir: Path, top_k: int = 10) -> TunerSummary:
        """Run auto-tuning and save the best models.

        Parameters
        ----------
        df : pd.DataFrame
            Bingo18 data.
        save_dir : Path
            Directory to save models.
        top_k : int
            Number of top results to save.

        Returns
        -------
        TunerSummary with results.
        """
        summary = self.run(df, top_k=top_k)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save top models
        for i, result in enumerate(summary.top_results[:top_k]):
            model = Bingo18Model(
                window=result.window,
                algorithm=result.algorithm,
                n_estimators=result.n_estimators,
                max_depth=result.max_depth,
            )
            model.train(df)

            model_name = f"model_{result.algorithm}_{result.bet_type}_{result.strategy}_{timestamp}_{i}.joblib"
            model_path = save_dir / model_name
            model.save(model_path)
            result.model_path = str(model_path)

        # Save best model metadata
        best_meta = {
            "timestamp": timestamp,
            "best_by_budget": {str(k): v.to_dict() for k, v in summary.best_by_budget.items()},
            "top_results": [r.to_dict() for r in summary.top_results],
        }
        meta_path = save_dir / "best.json"
        meta_path.write_text(json.dumps(best_meta, indent=2, ensure_ascii=False))
        logger.info(f"Best model metadata saved to {meta_path}")

        # Save full summary
        summary_path = save_dir / f"tuner_summary_{timestamp}.json"
        summary_path.write_text(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))
        logger.info(f"Tuner summary saved to {summary_path}")

        return summary


def render_tuner_results(summary: TunerSummary) -> str:
    """Render auto-tuner results as markdown."""
    lines = [
        "# Bingo18 Auto-Tuner Results",
        "",
        f"> Generated: {summary.timestamp}",
        "",
        "## Overview",
        "",
        f"- Total combinations tested: {summary.total_combinations}",
        f"- Total simulations: {summary.total_simulations}",
        f"- Budget levels: {', '.join(f'{b:,}' for b in summary.budget_levels)} VND",
        "",
    ]

    # Best per budget
    lines.append("## Best Strategy per Budget Level")
    lines.append("")

    headers = [
        "Budget",
        "Algorithm",
        "Window",
        "Est.",
        "Depth",
        "Bet Type",
        "Strategy",
        "Final Budget",
        "ROI",
        "Bets Survived",
        "Win Rate",
    ]
    rows = []
    for budget, best in sorted(summary.best_by_budget.items()):
        rows.append(
            [
                f"{budget:,}",
                best.algorithm,
                best.window,
                best.n_estimators or "-",
                best.max_depth or "-",
                best.bet_type,
                best.strategy,
                f"{best.final_budget:,}",
                f"{best.roi:.2f}%",
                f"{best.bets_survived:,}",
                f"{best.win_rate:.2%}",
            ]
        )
    lines.append(tabulate(rows, headers=headers, tablefmt="pipe"))
    lines.append("")

    # Top results
    lines.append("## Top Results (across all budgets)")
    lines.append("")

    top_headers = ["#", "Algorithm", "Bet Type", "Strategy", "Budget", "Final Budget", "ROI", "Bets", "Win Rate"]
    top_rows = []
    for i, r in enumerate(summary.top_results, 1):
        top_rows.append(
            [
                i,
                r.algorithm,
                r.bet_type,
                r.strategy,
                f"{r.budget:,}",
                f"{r.final_budget:,}",
                f"{r.roi:.2f}%",
                f"{r.bets_survived:,}",
                f"{r.win_rate:.2%}",
            ]
        )
    lines.append(tabulate(top_rows, headers=top_headers, tablefmt="pipe"))
    lines.append("")

    if summary.top_results and summary.top_results[0].model_path:
        lines.append("## Saved Models")
        lines.append("")
        for r in summary.top_results:
            if r.model_path:
                lines.append(f"- `{r.model_path}` ({r.algorithm} + {r.bet_type})")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Disclaimer: Lottery outcomes are random. These results are for educational/research purposes only.*")

    return "\n".join(lines)
