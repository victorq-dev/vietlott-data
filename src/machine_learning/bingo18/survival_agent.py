"""SurvivalAgent: absence-based portfolio betting agent for Bingo18.

Instead of predicting digit probabilities, this agent tracks how long each
bet option has been "absent" (not won) and bets on overdue outcomes.

Bet selection:
- absence_ratio = draws_since_win / expected_gap
- Bet when absence_ratio >= min_absence_ratio (default 1.0 = exactly at gap)
- Units per bet = min(floor(absence_ratio), max_units_per_option)
- Each unit = bet_size VND (default 10,000)

Goal: maximize survival draws, not ROI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from machine_learning.bingo18.simulator import BetType, calculate_payout

# ---------------------------------------------------------------------------
# Expected gaps (theoretical, fair 3d6 with faces 1-6)
# ---------------------------------------------------------------------------

# Number of ways to roll each total with 3d6
_CONG_TONG_WAYS: dict[int, int] = {
    3: 1,
    4: 3,
    5: 6,
    6: 10,
    7: 15,
    8: 21,
    9: 25,
    10: 27,
    11: 27,
    12: 25,
    13: 21,
    14: 15,
    15: 10,
    16: 6,
    17: 3,
    18: 1,
}


def _expected_gap(bet_type: BetType, value: Any) -> float:
    """Expected draws between consecutive wins for a bet option (fair 3d6)."""
    if bet_type == BetType.MOT_SO:
        return 216 / 91  # P(digit appears ≥1 time) = 91/216
    if bet_type == BetType.HAI_SO_TRUNG:
        return 216 / 16  # P(digit appears ≥2 times) = 16/216
    if bet_type == BetType.BA_SO_TRUNG:
        return 216.0  # P(digit appears 3 times) = 1/216
    if bet_type == BetType.CONG_TONG:
        ways = _CONG_TONG_WAYS.get(int(value), 1)
        return 216 / ways
    if bet_type == BetType.LON_HOA_NHO:
        if value in ("Nhỏ", "Lớn"):
            return 216 / 81  # P(Nhỏ or Lớn) = 81/216
        return 216 / 54  # P(Hòa) = 54/216
    return 10.0


def _all_bet_options() -> list[tuple[BetType, Any]]:
    opts: list[tuple[BetType, Any]] = []
    for d in range(1, 7):
        opts.append((BetType.MOT_SO, d))
        opts.append((BetType.HAI_SO_TRUNG, d))
        opts.append((BetType.BA_SO_TRUNG, d))
    for t in range(3, 19):
        opts.append((BetType.CONG_TONG, t))
    for cat in ["Nhỏ", "Hòa", "Lớn"]:
        opts.append((BetType.LON_HOA_NHO, cat))
    return opts


ALL_BET_OPTIONS: list[tuple[BetType, Any]] = _all_bet_options()


# ---------------------------------------------------------------------------
# BetOptionState
# ---------------------------------------------------------------------------


@dataclass
class BetOptionState:
    """Tracks absence and lifetime performance for one (bet_type, value) pair."""

    expected_gap: float
    draws_since_win: int = 0
    total_bets: int = 0
    total_units: int = 0
    total_wagered: int = 0
    total_payout: int = 0
    wins: int = 0

    @property
    def absence_ratio(self) -> float:
        """How overdue: 1.0 means exactly at expected gap, 2.0 means twice as long."""
        return self.draws_since_win / self.expected_gap

    @property
    def roi(self) -> float:
        if self.total_wagered == 0:
            return 0.0
        return (self.total_payout - self.total_wagered) / self.total_wagered * 100

    @property
    def win_rate(self) -> float:
        if self.total_bets == 0:
            return 0.0
        return self.wins / self.total_bets


# ---------------------------------------------------------------------------
# SurvivalAgent
# ---------------------------------------------------------------------------


class SurvivalAgent:
    """Absence-based portfolio betting agent for Bingo18.

    Tracks how long each bet option has been absent, then bets on overdue
    outcomes. Allocates more units (each = bet_size) to more-overdue options.
    """

    def __init__(
        self,
        agent_id: str,
        budget: int,
        bet_size: int = 10_000,
        available_bet_types: list[BetType] | None = None,
        max_units_per_draw: int = 5,
        max_units_per_option: int = 3,
        min_absence_ratio: float = 1.0,
    ) -> None:
        self.agent_id = agent_id
        self.budget = budget
        self.bet_size = bet_size
        self.max_units_per_draw = max_units_per_draw
        self.max_units_per_option = max_units_per_option
        self.min_absence_ratio = min_absence_ratio

        self._starting_budget = budget
        self._available_types: set[BetType] = set(available_bet_types or list(BetType))

        self._states: dict[tuple[BetType, Any], BetOptionState] = {
            (bt, val): BetOptionState(expected_gap=_expected_gap(bt, val))
            for bt, val in ALL_BET_OPTIONS
            if bt in self._available_types
        }

        self._total_draws: int = 0
        self._total_bets: int = 0
        self._wins: int = 0
        self._losses: int = 0
        self._profit_curve: list[int] = [budget]
        self._max_budget: int = budget
        self._min_budget: int = budget
        self._max_drawdown: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_alive(self) -> bool:
        return self.budget >= self.bet_size

    @property
    def roi(self) -> float:
        if self._starting_budget == 0:
            return 0.0
        return (self.budget - self._starting_budget) / self._starting_budget * 100

    @property
    def win_rate(self) -> float:
        if self._total_bets == 0:
            return 0.0
        return self._wins / self._total_bets

    @property
    def lifetime_roi(self) -> float:
        """ROI computed from all bets ever placed — not affected by bankruptcies."""
        total_wagered = sum(s.total_wagered for s in self._states.values())
        total_payout = sum(s.total_payout for s in self._states.values())
        if total_wagered == 0:
            return 0.0
        return (total_payout - total_wagered) / total_wagered * 100

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _select_portfolio(self) -> list[tuple[BetType, Any, int]]:
        """Select bets for this draw: returns list of (bet_type, value, units)."""
        candidates: list[tuple[float, BetType, Any]] = []
        for (bt, val), state in self._states.items():
            if state.absence_ratio >= self.min_absence_ratio:
                candidates.append((state.absence_ratio, bt, val))

        candidates.sort(reverse=True)  # most overdue first

        portfolio: list[tuple[BetType, Any, int]] = []
        total_units = 0

        for absence_ratio, bt, val in candidates:
            if total_units >= self.max_units_per_draw:
                break
            affordable = self.budget // self.bet_size - total_units
            if affordable <= 0:
                break
            units = min(
                int(absence_ratio),
                self.max_units_per_option,
                self.max_units_per_draw - total_units,
                affordable,
            )
            if units > 0:
                portfolio.append((bt, val, units))
                total_units += units

        return portfolio

    def process_draw(
        self,
        actual_digits: list[int],
        actual_total: int,
        large_small: str,
        date: str = "",
        draw_id: str = "",
    ) -> list[dict[str, Any]]:
        """Process one draw: select bets, record outcomes, update absence counters."""
        self._total_draws += 1
        portfolio = self._select_portfolio()
        outcomes: list[dict[str, Any]] = []

        for bt, val, units in portfolio:
            bet_amount = units * self.bet_size
            if self.budget < bet_amount:
                continue

            _matches, payout = calculate_payout(bt, val, actual_digits, bet_amount)
            self.budget -= bet_amount
            self.budget += payout

            state = self._states[(bt, val)]
            state.total_bets += 1
            state.total_units += units
            state.total_wagered += bet_amount
            state.total_payout += payout
            won = payout > 0
            if won:
                state.wins += 1
                self._wins += 1
            else:
                self._losses += 1
            self._total_bets += 1

            outcomes.append(
                {
                    "bet_type": bt.value,
                    "value": val,
                    "units": units,
                    "amount": bet_amount,
                    "payout": payout,
                    "won": won,
                }
            )

        # Update absence counters for ALL tracked options
        for (bt, val), state in self._states.items():
            if self._check_hit(bt, val, actual_digits, actual_total, large_small):
                state.draws_since_win = 0
            else:
                state.draws_since_win += 1

        # Budget curve and drawdown tracking
        self._profit_curve.append(self.budget)
        if self.budget > self._max_budget:
            self._max_budget = self.budget
        if self.budget < self._min_budget:
            self._min_budget = self.budget
        drawdown = self._max_budget - self.budget
        if drawdown > self._max_drawdown:
            self._max_drawdown = drawdown

        return outcomes

    @staticmethod
    def _check_hit(bt: BetType, val: Any, digits: list[int], total: int, ls: str) -> bool:
        """Check if a bet option would win this draw."""
        if bt == BetType.MOT_SO:
            return val in digits
        if bt == BetType.HAI_SO_TRUNG:
            return digits.count(val) >= 2
        if bt == BetType.BA_SO_TRUNG:
            return digits.count(val) == 3
        if bt == BetType.CONG_TONG:
            return total == val
        if bt == BetType.LON_HOA_NHO:
            return ls == val
        return False

    def reset_budget(self) -> None:
        """Reset budget to starting value — keeps all learned absence/stats."""
        logger.debug(f"[{self.agent_id}] Budget reset: {self.budget:,} → {self._starting_budget:,}")
        self.budget = self._starting_budget
        self._profit_curve.append(self.budget)
        self._max_budget = self.budget
        self._min_budget = self.budget

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save agent state to <path>/<agent_id>.json."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        filepath = path / f"{self.agent_id}.json"

        states_dict: dict[str, dict] = {}
        for (bt, val), s in self._states.items():
            key = f"{bt.value}:{val}"
            states_dict[key] = {
                "expected_gap": s.expected_gap,
                "draws_since_win": s.draws_since_win,
                "total_bets": s.total_bets,
                "total_units": s.total_units,
                "total_wagered": s.total_wagered,
                "total_payout": s.total_payout,
                "wins": s.wins,
            }

        data = {
            "agent_id": self.agent_id,
            "budget": self.budget,
            "bet_size": self.bet_size,
            "available_bet_types": [bt.value for bt in self._available_types],
            "max_units_per_draw": self.max_units_per_draw,
            "max_units_per_option": self.max_units_per_option,
            "min_absence_ratio": self.min_absence_ratio,
            "state": {
                "starting_budget": self._starting_budget,
                "total_draws": self._total_draws,
                "total_bets": self._total_bets,
                "wins": self._wins,
                "losses": self._losses,
                "profit_curve": self._profit_curve[-100:],
                "max_budget": self._max_budget,
                "min_budget": self._min_budget,
                "max_drawdown": self._max_drawdown,
                "option_states": states_dict,
            },
        }

        with filepath.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"[{self.agent_id}] Saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "SurvivalAgent":
        """Load agent state from a JSON file."""
        filepath = Path(filepath)
        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)

        available = [BetType(v) for v in data.get("available_bet_types", [bt.value for bt in BetType])]
        agent = cls(
            agent_id=data["agent_id"],
            budget=data["budget"],
            bet_size=data.get("bet_size", 10_000),
            available_bet_types=available,
            max_units_per_draw=data.get("max_units_per_draw", 5),
            max_units_per_option=data.get("max_units_per_option", 3),
            min_absence_ratio=data.get("min_absence_ratio", 1.0),
        )

        s = data["state"]
        agent._starting_budget = s["starting_budget"]
        agent._total_draws = s["total_draws"]
        agent._total_bets = s["total_bets"]
        agent._wins = s["wins"]
        agent._losses = s["losses"]
        agent._profit_curve = s["profit_curve"]
        agent._max_budget = s["max_budget"]
        agent._min_budget = s["min_budget"]
        agent._max_drawdown = s["max_drawdown"]

        for key_str, sd in s.get("option_states", {}).items():
            bt_str, val_str = key_str.split(":", 1)
            bt = BetType(bt_str)
            val: Any
            if bt in (BetType.MOT_SO, BetType.HAI_SO_TRUNG, BetType.BA_SO_TRUNG, BetType.CONG_TONG):
                val = int(val_str)
            else:
                val = val_str
            key = (bt, val)
            if key in agent._states:
                st = agent._states[key]
                st.draws_since_win = sd["draws_since_win"]
                st.total_bets = sd["total_bets"]
                st.total_units = sd.get("total_units", 0)
                st.total_wagered = sd["total_wagered"]
                st.total_payout = sd["total_payout"]
                st.wins = sd["wins"]

        logger.info(
            f"[{agent.agent_id}] Loaded: budget={agent.budget:,}, "
            f"draws={agent._total_draws}, lifetime_roi={agent.lifetime_roi:+.1f}%"
        )
        return agent
