"""EVAgent: Expected-Value-based betting agent for Bingo18.

Estimates per-digit probabilities from a rolling window of recent draws,
blends with theoretical fair-dice priors, computes EV for every bet option,
then selects a single dominant bet or a cross-region portfolio.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from loguru import logger

from machine_learning.bingo18.dice_probs import (
    compute_cong_tong_ev,
    compute_lon_hoa_nho_ev,
    compute_mot_so_ev,
    compute_pair_prob,
    compute_triple_prob,
)
from machine_learning.bingo18.simulator import (
    BA_SO_TRUNG_ANY_PRIZE,
    BA_SO_TRUNG_PRIZE,
    HAI_SO_TRUNG_PRIZE,
    BetType,
    calculate_payout,
)

_FAIR_DIGIT_PROBS: dict[int, float] = {d: 1.0 / 6.0 for d in range(1, 7)}

# All scorable (BetType, value) pairs — iterated in _select_bets
_ALL_OPTIONS: list[tuple[BetType, Any]] = (
    [(BetType.MOT_SO, d) for d in range(1, 7)]
    + [(BetType.HAI_SO_TRUNG, d) for d in range(1, 7)]
    + [(BetType.BA_SO_TRUNG, d) for d in range(1, 7)]
    + [(BetType.BA_SO_TRUNG_ANY, None)]
    + [(BetType.CONG_TONG, t) for t in range(3, 19)]
    + [(BetType.LON_HOA_NHO, cat) for cat in ["Nhỏ", "Hòa", "Lớn"]]
)

# Safe pool: only bets with win rate >= ~7% (excludes ba_so_trung, hai_so_trung, extreme cong_tong totals).
# ba_so_trung win rate = 0.46% → agent loses every bet → bankrupts in ~100 bets.
# This pool guarantees enough wins per 100 bets to sustain a budget over many draws.
_SAFE_OPTIONS: list[tuple[BetType, Any]] = (
    [(BetType.MOT_SO, d) for d in range(1, 7)]  # ~42% win rate
    + [(BetType.CONG_TONG, t) for t in range(7, 15)]  # 7-14: win rate 6.9-12.5%
    + [(BetType.LON_HOA_NHO, cat) for cat in ["Nhỏ", "Hòa", "Lớn"]]  # 25-37.5% win rate
)


def _option_region(bet_type: BetType, value: Any) -> str:
    """Map a bet option to a region string for portfolio deduplication.

    Bets in the same region cover overlapping outcomes; a portfolio should
    contain at most one bet per region.
    """
    if bet_type == BetType.LON_HOA_NHO:
        if value == "Nhỏ":
            return "nho"
        if value == "Lớn":
            return "lon"
        return "hoa"
    if bet_type == BetType.CONG_TONG:
        if value <= 9:
            return "nho"
        if value <= 11:
            return "hoa"
        return "lon"
    if bet_type in (BetType.MOT_SO, BetType.HAI_SO_TRUNG, BetType.BA_SO_TRUNG):
        return f"digit_{value}"
    return "any_triple"


def _compute_ev(bet_type: BetType, value: Any, digit_probs: dict[int, float]) -> float:
    """Compute expected value per 10k bet for a given option and digit distribution."""
    if bet_type == BetType.MOT_SO:
        return compute_mot_so_ev(digit_probs, value)
    if bet_type == BetType.HAI_SO_TRUNG:
        p = compute_pair_prob(digit_probs, value)
        return p * HAI_SO_TRUNG_PRIZE - 10_000.0
    if bet_type == BetType.BA_SO_TRUNG:
        p = compute_triple_prob(digit_probs, value)
        return p * BA_SO_TRUNG_PRIZE - 10_000.0
    if bet_type == BetType.BA_SO_TRUNG_ANY:
        p = sum(compute_triple_prob(digit_probs, d) for d in range(1, 7))
        return p * BA_SO_TRUNG_ANY_PRIZE - 10_000.0
    if bet_type == BetType.CONG_TONG:
        return compute_cong_tong_ev(digit_probs, value)
    if bet_type == BetType.LON_HOA_NHO:
        return compute_lon_hoa_nho_ev(digit_probs, value)
    raise ValueError(f"Unknown bet type: {bet_type}")


# Fair-dice EV baseline for each option — precomputed once.
# Used to compute edge = current_ev - fair_ev.
_FAIR_EVS: dict[tuple[BetType, Any], float] = {
    (bt, val): _compute_ev(bt, val, _FAIR_DIGIT_PROBS) for bt, val in _ALL_OPTIONS
}


class EVAgent:
    """Bingo18 agent that selects bets by expected value.

    Maintains a rolling window of recent draw history to estimate per-digit
    probabilities, blends them with a fair-dice prior (controlled by
    ``blend_alpha``), and computes EV for every bet option before each draw.

    Parameters
    ----------
    agent_id : str
    budget : int — starting bankroll in VND
    bet_size : int — cost per betting unit (default 10 000 VND)
    window : int — max recent draws to retain for probability estimation
    min_ev_per_10k : float — absolute EV floor; used when min_edge_over_fair is None
    min_edge_over_fair : float | None — if set, bet only when current_ev - fair_ev >= this
        value.  Positive = requires genuine edge over fair dice.  0 = any improvement
        over fair.  None = use min_ev_per_10k instead (legacy mode).
    blend_alpha : float — weight on the fair prior (0 = pure empirical, 1 = always fair)
    max_bets_per_draw : int — portfolio size cap
    single_ev_gap : float — if best_ev - second_ev >= gap, place only the single best bet
    """

    def __init__(
        self,
        agent_id: str,
        budget: int,
        bet_size: int = 10_000,
        window: int = 100,
        min_ev_per_10k: float = -4_200.0,
        min_edge_over_fair: float | None = None,
        blend_alpha: float = 0.5,
        max_bets_per_draw: int = 2,
        single_ev_gap: float = 200.0,
        safe_bets_only: bool = False,
    ) -> None:
        self.agent_id = agent_id
        self.budget = budget
        self.bet_size = bet_size
        self.window = window
        self.min_ev_per_10k = min_ev_per_10k
        self.min_edge_over_fair = min_edge_over_fair
        self.blend_alpha = blend_alpha
        self.max_bets_per_draw = max_bets_per_draw
        self.single_ev_gap = single_ev_gap
        self.safe_bets_only = safe_bets_only

        self._initial_budget: int = budget
        self._recent_draws: deque[list[int]] = deque(maxlen=window)
        self._total_draws: int = 0
        self._total_bets: int = 0
        self._total_wins: int = 0
        self._total_wagered: int = 0
        self._total_payout: int = 0

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def warm_up(self, draws: list[list[int]]) -> None:
        """Pre-populate the rolling window with historical draws."""
        for draw in draws:
            self._recent_draws.append(draw)

    # ------------------------------------------------------------------
    # Probability estimation
    # ------------------------------------------------------------------

    def _estimate_digit_probs(self) -> dict[int, float]:
        """Return blended per-digit probabilities from recent history.

        With an empty buffer returns fair-dice priors.
        """
        if not self._recent_draws:
            return dict(_FAIR_DIGIT_PROBS)

        counts: dict[int, int] = {d: 0 for d in range(1, 7)}
        for draw in self._recent_draws:
            for digit in draw:
                if digit in counts:
                    counts[digit] += 1

        total_dice = len(self._recent_draws) * 3
        empirical = {d: counts[d] / total_dice for d in range(1, 7)}

        return {d: self.blend_alpha * (1.0 / 6.0) + (1.0 - self.blend_alpha) * empirical[d] for d in range(1, 7)}

    # ------------------------------------------------------------------
    # Bet selection
    # ------------------------------------------------------------------

    def _select_bets(self, digit_probs: dict[int, float]) -> list[dict]:
        """Select bets for the next draw given estimated digit probabilities.

        Returns a list of bet dicts with keys: bet_type, value, units, ev.
        """
        options = _SAFE_OPTIONS if self.safe_bets_only else _ALL_OPTIONS
        scored = [
            {
                "bet_type": bt,
                "value": val,
                "ev": _compute_ev(bt, val, digit_probs),
                "units": 1,
            }
            for bt, val in options
        ]

        if self.min_edge_over_fair is not None:
            # Edge-over-fair mode: skip draw when no option beats fair EV by threshold.
            # min_edge_over_fair=0 = bet only when any option is above fair-dice baseline.
            passing = [b for b in scored if b["ev"] - _FAIR_EVS[(b["bet_type"], b["value"])] >= self.min_edge_over_fair]
        else:
            passing = [b for b in scored if b["ev"] >= self.min_ev_per_10k]
        if not passing:
            return []

        passing.sort(key=lambda x: x["ev"], reverse=True)

        # Single dominant bet when the gap to the runner-up is large enough
        if len(passing) == 1 or (passing[0]["ev"] - passing[1]["ev"]) >= self.single_ev_gap:
            return [passing[0]]

        # Portfolio: one best option per region, up to max_bets_per_draw
        used_regions: set[str] = set()
        portfolio: list[dict] = []
        for bet in passing:
            if len(portfolio) >= self.max_bets_per_draw:
                break
            region = _option_region(bet["bet_type"], bet["value"])
            if region not in used_regions:
                used_regions.add(region)
                portfolio.append(bet)

        return portfolio

    # ------------------------------------------------------------------
    # Main per-draw entry point
    # ------------------------------------------------------------------

    def process_draw(
        self,
        actual_digits: list[int],
        actual_total: int,
        large_small: str,
        date: str,
        draw_id: str,
    ) -> list[dict]:
        """Decide bets, evaluate outcomes, and update internal state.

        Returns a list of outcome dicts (empty when skipping).
        """
        if not self.is_alive:
            self._total_draws += 1
            self._recent_draws.append(list(actual_digits))
            return []

        digit_probs = self._estimate_digit_probs()
        selected = self._select_bets(digit_probs)

        outcomes: list[dict] = []
        for bet in selected:
            units = bet["units"]
            amount = self.bet_size * units
            _, payout = calculate_payout(bet["bet_type"], bet["value"], actual_digits, amount)
            won = payout > 0

            self.budget += payout - amount
            self._total_wagered += amount
            self._total_payout += payout
            if won:
                self._total_wins += 1

            outcomes.append(
                {
                    "bet_type": bet["bet_type"],
                    "value": bet["value"],
                    "units": units,
                    "amount": amount,
                    "payout": payout,
                    "won": won,
                    "ev": bet["ev"],
                }
            )

        self._total_bets += len(outcomes)
        self._total_draws += 1
        self._recent_draws.append(list(actual_digits))
        return outcomes

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def lifetime_roi(self) -> float:
        """Net profit as a percentage of total amount wagered."""
        if self._total_wagered == 0:
            return 0.0
        return (self._total_payout - self._total_wagered) / self._total_wagered * 100

    @property
    def roi(self) -> float:
        """Alias for lifetime_roi."""
        return self.lifetime_roi

    @property
    def win_rate(self) -> float:
        """Fraction of individual bets that produced a non-zero payout."""
        if self._total_bets == 0:
            return 0.0
        return self._total_wins / self._total_bets

    @property
    def is_alive(self) -> bool:
        """True when remaining budget can cover at least one bet."""
        return self.budget >= self.bet_size

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: Path) -> None:
        """Serialise agent state to JSON at ``directory/{agent_id}.json``."""
        data = {
            "agent_id": self.agent_id,
            "budget": self.budget,
            "bet_size": self.bet_size,
            "window": self.window,
            "min_ev_per_10k": self.min_ev_per_10k,
            "min_edge_over_fair": self.min_edge_over_fair,
            "blend_alpha": self.blend_alpha,
            "max_bets_per_draw": self.max_bets_per_draw,
            "single_ev_gap": self.single_ev_gap,
            "safe_bets_only": self.safe_bets_only,
            "_initial_budget": self._initial_budget,
            "_total_draws": self._total_draws,
            "_total_bets": self._total_bets,
            "_total_wins": self._total_wins,
            "_total_wagered": self._total_wagered,
            "_total_payout": self._total_payout,
            "_recent_draws": list(self._recent_draws),
        }
        path = directory / f"{self.agent_id}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        logger.info(f"EVAgent {self.agent_id!r} saved to {path}")

    @classmethod
    def load(cls, path: Path) -> EVAgent:
        """Deserialise an EVAgent from a JSON file produced by :meth:`save`."""
        data = json.loads(path.read_text())
        agent = cls(
            agent_id=data["agent_id"],
            budget=data["budget"],
            bet_size=data["bet_size"],
            window=data["window"],
            min_ev_per_10k=data["min_ev_per_10k"],
            min_edge_over_fair=data.get("min_edge_over_fair"),
            blend_alpha=data["blend_alpha"],
            max_bets_per_draw=data["max_bets_per_draw"],
            single_ev_gap=data["single_ev_gap"],
            safe_bets_only=data.get("safe_bets_only", False),
        )
        agent._initial_budget = data["_initial_budget"]
        agent._total_draws = data["_total_draws"]
        agent._total_bets = data["_total_bets"]
        agent._total_wins = data["_total_wins"]
        agent._total_wagered = data["_total_wagered"]
        agent._total_payout = data["_total_payout"]
        for draw in data["_recent_draws"]:
            agent._recent_draws.append(draw)
        logger.info(f"EVAgent {agent.agent_id!r} loaded from {path}")
        return agent
