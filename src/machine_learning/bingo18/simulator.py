"""Auto-play simulation engine for Bingo18.

Supports all Bingo18 bet types based on official rules:
- "mot_so": Pick 1 number (1-6), win if it appears in the draw
- "hai_so_trung": Pick a number, win if it appears at least twice
- "ba_so_trung": Pick a number, win if all 3 draws are that number
- "cong_tong": Pick a total (3-18), win if sum matches
- "lon_hoa_nho": Pick Big/Draw/Small, win if total matches range
- "cong_tong_mult": Pick total sum with multiplier-based payout
- "lon_hoa_nho_v2": Big/Draw/Small with multiplier-based payout
- "trung_2so": Any pair appears, x7.5
- "trung_3so": Specific triple, x120
- "trung_3so_any": Specific digit triple, x20
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from machine_learning.bingo18.model import Bingo18Model


class BettingMode(str, Enum):
    """How to place bets across bet types."""

    SINGLE = "single"  # One bet type per draw (original behavior)
    COMBINE = "combine"  # Multiple bet types per draw
    ALL_IN = "all_in"  # All budget on best single bet type
    SKIP = "skip"  # Skip draws below confidence threshold


class BetType(str, Enum):
    """Bingo18 bet types."""

    MOT_SO = "mot_so"  # Pick 1 number
    HAI_SO_TRUNG = "hai_so_trung"  # Pick number, win if appears 2+ times
    BA_SO_TRUNG = "ba_so_trung"  # Pick number, win if appears 3 times
    CONG_TONG = "cong_tong"  # Pick total sum
    LON_HOA_NHO = "lon_hoa_nho"  # Pick Big/Draw/Small
    CONG_TONG_MULT = "cong_tong_mult"  # Pick total sum, multiplier-based
    LON_HOA_NHO_V2 = "lon_hoa_nho_v2"  # Big/Draw/Small, multiplier-based
    TRUNG_2SO = "trung_2so"  # Specific digit pair (must pick digit 1-6)
    TRUNG_3SO = "trung_3so"  # Specific triple
    TRUNG_3SO_ANY = "trung_3so_any"  # Specific digit triple (must pick digit 1-6)


# Prize table for "Một số" (One number) bet
# Key: count of matching digits -> prize in VND
MOT_SO_PRIZE = {0: 0, 1: 12_000, 2: 20_000, 3: 30_000}

# Prize table for "Cộng tổng" (Sum) bet
CONG_TONG_PRIZE = {
    3: 1_200_000,
    4: 400_000,
    5: 200_000,
    6: 120_000,
    7: 80_000,
    8: 55_000,
    9: 47_000,
    10: 44_000,
    11: 44_000,
    12: 47_000,
    13: 55_000,
    14: 80_000,
    15: 120_000,
    16: 200_000,
    17: 400_000,
    18: 1_200_000,
}

# Prize table for "Lớn/Hòa/Nhỏ" (Big/Draw/Small) bet
LON_HOA_NHO_PRIZE = {
    "Nhỏ": 15_000,  # Total 3-9
    "Hòa": 20_000,  # Total 10-11
    "Lớn": 15_000,  # Total 12-18
}

# Fixed prizes
HAI_SO_TRUNG_PRIZE = 75_000  # At least 2 matching digits
BA_SO_TRUNG_PRIZE = 1_200_000  # All 3 digits match (specific number)
BA_SO_TRUNG_ANY_PRIZE = 200_000  # All 3 digits match (any triple)

# Multiplier-based prize tables (payout = bet_size * multiplier)
CONG_TONG_MULTIPLIER = {
    3: 120.0,
    4: 40.0,
    5: 20.0,
    6: 12.0,
    7: 8.0,
    8: 5.5,
    9: 4.7,
    10: 4.4,
    11: 4.4,
    12: 4.7,
    13: 5.5,
    14: 8.0,
    15: 12.0,
    16: 20.0,
    17: 40.0,
    18: 120.0,
}

LON_HOA_NHO_V2_MULTIPLIER = {
    "Nhỏ": 1.5,  # Total 3-9
    "Hòa": 2.0,  # Total 10-11
    "Lớn": 1.5,  # Total 12-18
}

TRUNG_2SO_MULTIPLIER = 7.5
TRUNG_3SO_MULTIPLIER = 120.0
TRUNG_3SO_ANY_MULTIPLIER = 20.0


@dataclass
class BetRecord:
    """Record of a single bet."""

    date: str
    draw_id: str
    bet_type: str
    bet_value: Any  # digit for mot_so, total for cong_tong, etc.
    actual_digits: list[int]
    actual_total: int
    matches: int
    bet_amount: int
    payout: int
    budget_after: int


@dataclass
class SimulationResult:
    """Complete result of a simulation run."""

    starting_budget: int
    final_budget: int
    bet_size: int
    bet_type: str
    total_bets: int = 0
    wins: int = 0
    losses: int = 0
    max_budget: int = 0
    min_budget: int = 0
    max_drawdown: int = 0
    bet_history: list[BetRecord] = field(default_factory=list)
    profit_curve: list[int] = field(default_factory=list)

    @property
    def profit(self) -> int:
        return self.final_budget - self.starting_budget

    @property
    def roi(self) -> float:
        if self.starting_budget == 0:
            return 0.0
        return (self.profit / self.starting_budget) * 100

    @property
    def win_rate(self) -> float:
        if self.total_bets == 0:
            return 0.0
        return self.wins / self.total_bets

    def to_dict(self) -> dict[str, Any]:
        return {
            "starting_budget": self.starting_budget,
            "final_budget": self.final_budget,
            "profit": self.profit,
            "roi_pct": round(self.roi, 2),
            "bet_size": self.bet_size,
            "bet_type": self.bet_type,
            "total_bets": self.total_bets,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "max_drawdown": self.max_drawdown,
        }


def calculate_payout(bet_type: BetType, bet_value: Any, actual_digits: list[int], bet_size: int) -> tuple[int, int]:
    """Calculate payout for a bet.

    Parameters
    ----------
    bet_type : BetType
    bet_value : Any — the bet choice (digit, total, or category)
    actual_digits : list[int] — the 3 drawn digits
    bet_size : int — amount bet in VND

    Returns
    -------
    (matches, payout) tuple
    """
    actual_digits = list(actual_digits)
    actual_total = sum(actual_digits)

    if bet_type == BetType.MOT_SO:
        # Bet on a single number, win for each occurrence
        count = actual_digits.count(bet_value)
        payout = MOT_SO_PRIZE.get(count, 0)
        if payout > 0:
            payout = int(payout * (bet_size / 10_000))
        return count, payout

    elif bet_type == BetType.HAI_SO_TRUNG:
        # Bet on a number appearing at least twice
        count = actual_digits.count(bet_value)
        if count >= 2:
            payout = int(HAI_SO_TRUNG_PRIZE * (bet_size / 10_000))
            return count, payout
        return count, 0

    elif bet_type == BetType.BA_SO_TRUNG:
        # Bet on a number appearing 3 times
        count = actual_digits.count(bet_value)
        if count == 3:
            payout = int(BA_SO_TRUNG_PRIZE * (bet_size / 10_000))
            return count, payout
        return count, 0

    elif bet_type == BetType.CONG_TONG:
        # Bet on total sum
        if actual_total == bet_value:
            prize = CONG_TONG_PRIZE.get(bet_value, 0)
            payout = int(prize * (bet_size / 10_000))
            return 1, payout
        return 0, 0

    elif bet_type == BetType.LON_HOA_NHO:
        # Bet on Big/Draw/Small
        if actual_total <= 9:
            category = "Nhỏ"
        elif actual_total <= 11:
            category = "Hòa"
        else:
            category = "Lớn"

        if category == bet_value:
            prize = LON_HOA_NHO_PRIZE.get(bet_value, 0)
            payout = int(prize * (bet_size / 10_000))
            return 1, payout
        return 0, 0

    elif bet_type == BetType.CONG_TONG_MULT:
        # Bet on total sum, multiplier-based
        if actual_total == bet_value:
            mult = CONG_TONG_MULTIPLIER.get(bet_value, 0)
            return 1, int(bet_size * mult)
        return 0, 0

    elif bet_type == BetType.LON_HOA_NHO_V2:
        # Big/Draw/Small, multiplier-based
        if actual_total <= 9:
            category = "Nhỏ"
        elif actual_total <= 11:
            category = "Hòa"
        else:
            category = "Lớn"

        if category == bet_value:
            mult = LON_HOA_NHO_V2_MULTIPLIER.get(bet_value, 0)
            return 1, int(bet_size * mult)
        return 0, 0

    elif bet_type == BetType.TRUNG_2SO:
        # Specific digit pair - must pick digit 1-6, win if it appears 2+
        count = actual_digits.count(bet_value)
        if count >= 2:
            return count, int(bet_size * TRUNG_2SO_MULTIPLIER)
        return 0, 0

    elif bet_type == BetType.TRUNG_3SO:
        # Specific triple
        count = actual_digits.count(bet_value)
        if count == 3:
            return 3, int(bet_size * TRUNG_3SO_MULTIPLIER)
        return count, 0

    elif bet_type == BetType.TRUNG_3SO_ANY:
        # Specific digit triple - must pick digit 1-6, win if all 3 match
        count = actual_digits.count(bet_value)
        if count >= 3:
            return count, int(bet_size * TRUNG_3SO_ANY_MULTIPLIER)
        return 0, 0

    else:
        raise ValueError(f"Unknown bet type: {bet_type}")


class Bingo18Simulator:
    """Simulate auto-play on Bingo18 with a trained model.

    For each draw:
    1. Use model to predict digit probabilities
    2. Select bet based on bet_type and strategy
    3. Deduct bet from budget
    4. Compare with actual result, calculate payout
    5. Stop if budget <= 0
    """

    def __init__(
        self,
        model: Bingo18Model,
        budget: int,
        bet_size: int = 10_000,
        bet_type: str = "mot_so",
        strategy: str = "top_n",
        top_n: int = 1,
        threshold: float = 0.12,
        target_total: int | None = None,
        target_category: str | None = None,
    ):
        self.model = model
        self.budget = budget
        self.bet_size = bet_size
        self.bet_type = BetType(bet_type)
        self.strategy = strategy
        self.top_n = top_n
        self.threshold = threshold
        self.target_total = target_total
        self.target_category = target_category

    def run(self, df: pd.DataFrame) -> SimulationResult:
        """Run simulation on historical data.

        Parameters
        ----------
        df : pd.DataFrame
            Full Bingo18 data. Simulation starts after model.window draws.

        Returns
        -------
        SimulationResult with full history.
        """
        window = self.model.window
        results = df["result"].tolist()
        totals = df["total"].tolist()
        large_small = df["large_small"].tolist()
        dates = df["date"].tolist()
        ids = df["id"].tolist() if "id" in df.columns else [str(i) for i in range(len(df))]

        budget = self.budget
        max_budget = budget
        max_drawdown = 0
        total_bets = 0
        wins = 0
        losses = 0
        bet_history = []
        profit_curve = [budget]

        start_idx = window
        logger.info(
            f"Starting simulation from draw {start_idx} with budget {budget:,} VND, bet_type={self.bet_type.value}"
        )

        for i in range(start_idx, len(results)):
            if budget < self.bet_size:
                logger.info(f"Budget exhausted at draw {i}. Remaining: {budget:,} VND")
                break

            # Build features from previous draws
            recent_draws = results[i - window : i]
            recent_totals = totals[i - window : i]
            recent_ls = large_small[i - window : i]

            X = self.model.feature_engineer.build_features_for_predict(recent_draws, recent_totals, recent_ls)

            # Select bet based on bet_type and strategy
            bet_value = self._select_bet(X)

            # Place bet
            budget -= self.bet_size
            total_bets += 1

            # Calculate payout
            matches, payout = calculate_payout(self.bet_type, bet_value, results[i], self.bet_size)
            budget += payout

            if payout > 0:
                wins += 1
            else:
                losses += 1

            # Track max drawdown
            if budget > max_budget:
                max_budget = budget
            drawdown = max_budget - budget
            if drawdown > max_drawdown:
                max_drawdown = drawdown

            profit_curve.append(budget)

            record = BetRecord(
                date=str(dates[i]),
                draw_id=str(ids[i]),
                bet_type=self.bet_type.value,
                bet_value=bet_value,
                actual_digits=results[i],
                actual_total=totals[i],
                matches=matches,
                bet_amount=self.bet_size,
                payout=payout,
                budget_after=budget,
            )
            bet_history.append(record)

        result = SimulationResult(
            starting_budget=self.budget,
            final_budget=budget,
            bet_size=self.bet_size,
            bet_type=self.bet_type.value,
            total_bets=total_bets,
            wins=wins,
            losses=losses,
            max_budget=max_budget,
            min_budget=min(budget, min(profit_curve)),
            max_drawdown=max_drawdown,
            bet_history=bet_history,
            profit_curve=profit_curve,
        )

        logger.info(f"Simulation complete: {total_bets} bets, profit={result.profit:,} VND, ROI={result.roi:.2f}%")
        return result

    def _select_bet(self, X: np.ndarray) -> Any:
        """Select bet value based on bet_type and strategy."""
        probs = self.model.predict_proba(X)

        if self.bet_type in (BetType.MOT_SO, BetType.HAI_SO_TRUNG, BetType.BA_SO_TRUNG, BetType.TRUNG_3SO, BetType.TRUNG_2SO):
            return self._select_digit(probs)

        elif self.bet_type in (BetType.CONG_TONG, BetType.CONG_TONG_MULT):
            if self.bet_type == BetType.CONG_TONG_MULT and self.model.total_clf is not None:
                return self._select_total_ml(X)
            return self._select_total(probs)

        elif self.bet_type in (BetType.LON_HOA_NHO, BetType.LON_HOA_NHO_V2):
            if self.bet_type == BetType.LON_HOA_NHO_V2 and self.model.total_clf is not None:
                return self._select_category_ml(X)
            return self._select_category(probs)

        elif self.bet_type == BetType.TRUNG_3SO_ANY:
            return self._select_digit(probs)

        else:
            raise ValueError(f"Unknown bet type: {self.bet_type}")

    def _select_total_ml(self, X: np.ndarray) -> int:
        """Select total using ML total classifier."""
        total_probs = self.model.predict_total_proba(X)
        if self.target_total is not None:
            return self.target_total
        return max(total_probs, key=lambda t: total_probs[t])

    def _select_category_ml(self, X: np.ndarray) -> str:
        """Select category using ML total classifier."""
        if self.target_category is not None:
            return self.target_category
        total_probs = self.model.predict_total_proba(X)
        p_small = sum(total_probs.get(t, 0) for t in range(3, 10))
        p_draw = sum(total_probs.get(t, 0) for t in [10, 11])
        p_big = sum(total_probs.get(t, 0) for t in range(12, 19))
        cat_probs = {"Nhỏ": p_small, "Hòa": p_draw, "Lớn": p_big}
        return max(cat_probs, key=cat_probs.get)

    def _select_digit(self, probs: dict[int, float]) -> int:
        """Select digit to bet on."""
        if self.strategy == "top_n":
            sorted_digits = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            return sorted_digits[0][0]  # Best digit
        elif self.strategy == "threshold":
            above = [d for d, p in probs.items() if p >= self.threshold]
            if above:
                return max(above, key=lambda d: probs[d])
            return max(probs, key=lambda d: probs[d])
        elif self.strategy == "kelly":
            return self._kelly_best_digit(probs)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _select_total(self, probs: dict[int, float]) -> int:
        """Select total sum to bet on.

        Uses model's digit probabilities to estimate which totals are most likely.
        Higher probability digits contribute to certain totals.
        """
        if self.target_total is not None:
            return self.target_total

        # Estimate total probability by summing contributions
        # A digit with higher probability is more likely to appear
        total_probs = {}
        for t in range(3, 19):
            # Probability of getting total t is complex, use approximation
            # based on digit probabilities
            total_probs[t] = 0.0

        # Simple approximation: use average of top digit probabilities
        sorted_probs = sorted(probs.values(), reverse=True)
        avg_top = np.mean(sorted_probs[:3])

        # Higher totals need higher digits, lower totals need lower digits
        # Map digit probabilities to total preferences
        for t in range(3, 19):
            # Distance from middle (10.5) - extreme totals are rarer but pay more
            mid_distance = abs(t - 10.5)
            # Weight by how likely the contributing digits are
            if t <= 9:
                # Small totals: favor low digits
                low_digit_prob = np.mean([probs.get(d, 0) for d in [1, 2, 3]])
                total_probs[t] = low_digit_prob * (1 + mid_distance * 0.1)
            elif t <= 11:
                # Middle totals: balanced
                total_probs[t] = avg_top
            else:
                # Large totals: favor high digits
                high_digit_prob = np.mean([probs.get(d, 0) for d in [4, 5, 6]])
                total_probs[t] = high_digit_prob * (1 + mid_distance * 0.1)

        if self.strategy == "top_n":
            return max(total_probs, key=lambda t: total_probs[t])
        elif self.strategy == "threshold":
            above = [t for t, p in total_probs.items() if p >= self.threshold]
            if above:
                return max(above, key=lambda t: total_probs[t])
            return max(total_probs, key=lambda t: total_probs[t])
        else:
            return max(total_probs, key=lambda t: total_probs[t])

    def _select_category(self, probs: dict[int, float]) -> str:
        """Select Big/Draw/Small category.

        Uses digit probabilities to estimate total distribution.
        """
        if self.target_category is not None:
            return self.target_category

        # Estimate probability of each category based on digit probs
        low_prob = np.mean([probs.get(d, 0) for d in [1, 2, 3]])
        high_prob = np.mean([probs.get(d, 0) for d in [4, 5, 6]])

        # Small (3-9): more low digits
        p_small = low_prob * 1.5
        # Big (12-18): more high digits
        p_big = high_prob * 1.5
        # Draw (10-11): balanced
        p_draw = (low_prob + high_prob) / 2

        category_probs = {"Nhỏ": p_small, "Hòa": p_draw, "Lớn": p_big}

        if self.strategy == "top_n":
            return max(category_probs, key=lambda c: category_probs[c])
        elif self.strategy == "threshold":
            above = [c for c, p in category_probs.items() if p >= self.threshold]
            if above:
                return max(above, key=lambda c: category_probs[c])
            return max(category_probs, key=lambda c: category_probs[c])
        else:
            return max(category_probs, key=lambda c: category_probs[c])

    def _kelly_best_digit(self, probs: dict[int, float]) -> int:
        """Select digit using Kelly criterion for 'mot_so' bet.

        Kelly fraction = (p * b - q) / b
        where p = win prob, b = net odds, q = 1 - p
        """
        best_digit = max(probs, key=lambda d: probs[d])

        for d, p in probs.items():
            # Expected value for mot_so bet:
            # P(match 1) * 12000 + P(match 2) * 20000 + P(match 3) * 30000
            # Simplified: use p as proxy for expected value
            b = 1.2  # net odds for 12k win on 10k bet
            q = 1 - p
            kelly = (p * b - q) / b if b > 0 else 0
            if kelly > 0:
                return d

        return best_digit

    def run_combined(
        self,
        df: pd.DataFrame,
        bet_types: list[str],
        mode: str = "combine",
        confidence_threshold: float = 0.0,
    ) -> SimulationResult:
        """Run simulation with multiple bet types per draw.

        Parameters
        ----------
        df : pd.DataFrame
            Full Bingo18 data.
        bet_types : list[str]
            Bet types to consider.
        mode : str
            "combine" = place multiple bets per draw, budget split evenly
            "all_in" = place all budget on single best bet
            "skip" = skip draws below confidence threshold
        confidence_threshold : float
            Minimum probability to place a bet (for "skip" mode).
        """
        window = self.model.window
        results = df["result"].tolist()
        totals = df["total"].tolist()
        large_small = df["large_small"].tolist()
        dates = df["date"].tolist()
        ids = df["id"].tolist() if "id" in df.columns else [str(i) for i in range(len(df))]

        budget = self.budget
        max_budget = budget
        max_drawdown = 0
        total_bets = 0
        wins = 0
        losses = 0
        bet_history = []
        profit_curve = [budget]
        bet_type_enums = [BetType(bt) for bt in bet_types]

        start_idx = window
        logger.info(
            f"Starting combined simulation from draw {start_idx}, "
            f"budget={budget:,} VND, mode={mode}, bet_types={bet_types}"
        )

        for i in range(start_idx, len(results)):
            if budget < self.bet_size:
                logger.info(f"Budget exhausted at draw {i}. Remaining: {budget:,} VND")
                break

            recent_draws = results[i - window : i]
            recent_totals = totals[i - window : i]
            recent_ls = large_small[i - window : i]
            X = self.model.feature_engineer.build_features_for_predict(recent_draws, recent_totals, recent_ls)

            # Score all bet types
            scored_bets = self._score_bets(X, bet_type_enums)

            if mode == "skip":
                # Only bet if best score above threshold
                if not scored_bets or scored_bets[0][0] < confidence_threshold:
                    profit_curve.append(budget)
                    continue

            if mode == "all_in":
                # Place all bet_size on the single best bet
                scored_bets = scored_bets[:1]

            elif mode == "combine":
                # Place bets on all positive-score bets, budget split evenly
                scored_bets = [(s, bt, bv) for s, bt, bv in scored_bets if s > 0]

            if not scored_bets:
                profit_curve.append(budget)
                continue

            # Calculate per-bet budget
            n_bets = len(scored_bets)
            per_bet = self.bet_size  # Each bet costs bet_size

            if budget < per_bet * n_bets:
                # Not enough for all bets, scale down
                n_bets = max(1, budget // per_bet)
                scored_bets = scored_bets[:n_bets]

            for _score, bt, bv in scored_bets:
                if budget < per_bet:
                    break
                budget -= per_bet
                total_bets += 1

                matches, payout = calculate_payout(bt, bv, results[i], per_bet)
                budget += payout

                if payout > 0:
                    wins += 1
                else:
                    losses += 1

                record = BetRecord(
                    date=str(dates[i]),
                    draw_id=str(ids[i]),
                    bet_type=bt.value,
                    bet_value=bv,
                    actual_digits=results[i],
                    actual_total=totals[i],
                    matches=matches,
                    bet_amount=per_bet,
                    payout=payout,
                    budget_after=budget,
                )
                bet_history.append(record)

            if budget > max_budget:
                max_budget = budget
            drawdown = max_budget - budget
            if drawdown > max_drawdown:
                max_drawdown = drawdown

            profit_curve.append(budget)

        result = SimulationResult(
            starting_budget=self.budget,
            final_budget=budget,
            bet_size=self.bet_size,
            bet_type="+".join(bet_types),
            total_bets=total_bets,
            wins=wins,
            losses=losses,
            max_budget=max_budget,
            min_budget=min(budget, min(profit_curve)),
            max_drawdown=max_drawdown,
            bet_history=bet_history,
            profit_curve=profit_curve,
        )

        logger.info(
            f"Combined simulation complete: {total_bets} bets, profit={result.profit:,} VND, ROI={result.roi:.2f}%"
        )
        return result

    def _score_bets(self, X: np.ndarray, bet_types: list[BetType]) -> list[tuple[float, BetType, Any]]:
        """Score all bet types by expected value. Returns sorted list of (score, bet_type, bet_value)."""
        probs = self.model.predict_proba(X)
        scored = []

        for bt in bet_types:
            if bt in (BetType.MOT_SO, BetType.HAI_SO_TRUNG, BetType.BA_SO_TRUNG):
                for d, p in probs.items():
                    ev = self._ev_digit(bt, d, p)
                    scored.append((ev, bt, d))

            elif bt == BetType.CONG_TONG:
                for t in range(3, 19):
                    p = self._estimate_total_prob(probs, t)
                    ev = p * CONG_TONG_PRIZE.get(t, 0) / 10_000
                    scored.append((ev, bt, t))

            elif bt == BetType.CONG_TONG_MULT:
                if self.model.total_clf is not None:
                    total_probs = self.model.predict_total_proba(X)
                    for t, p in total_probs.items():
                        ev = p * CONG_TONG_MULTIPLIER.get(t, 0)
                        scored.append((ev, bt, t))
                else:
                    for t in range(3, 19):
                        p = self._estimate_total_prob(probs, t)
                        ev = p * CONG_TONG_MULTIPLIER.get(t, 0)
                        scored.append((ev, bt, t))

            elif bt == BetType.LON_HOA_NHO:
                cat_probs = self._estimate_category_probs(probs)
                for cat, p in cat_probs.items():
                    ev = p * LON_HOA_NHO_PRIZE.get(cat, 0) / 10_000
                    scored.append((ev, bt, cat))

            elif bt == BetType.LON_HOA_NHO_V2:
                if self.model.total_clf is not None:
                    total_probs = self.model.predict_total_proba(X)
                    p_small = sum(total_probs.get(t, 0) for t in range(3, 10))
                    p_draw = sum(total_probs.get(t, 0) for t in [10, 11])
                    p_big = sum(total_probs.get(t, 0) for t in range(12, 19))
                else:
                    cat_probs = self._estimate_category_probs(probs)
                    p_small, p_draw, p_big = cat_probs["Nhỏ"], cat_probs["Hòa"], cat_probs["Lớn"]
                for cat, p in [("Nhỏ", p_small), ("Hòa", p_draw), ("Lớn", p_big)]:
                    ev = p * LON_HOA_NHO_V2_MULTIPLIER.get(cat, 0)
                    scored.append((ev, bt, cat))

            elif bt == BetType.TRUNG_2SO:
                for d in range(1, 7):
                    p = probs.get(d, 0)
                    p_pair = 3 * p * p * (1 - p) + p * p * p
                    ev = p_pair * TRUNG_2SO_MULTIPLIER
                    scored.append((ev, bt, d))

            elif bt == BetType.TRUNG_3SO:
                for d in range(1, 7):
                    p = probs.get(d, 0) ** 3
                    ev = p * TRUNG_3SO_MULTIPLIER
                    scored.append((ev, bt, d))

            elif bt == BetType.TRUNG_3SO_ANY:
                for d in range(1, 7):
                    p = probs.get(d, 0) ** 3
                    ev = p * TRUNG_3SO_ANY_MULTIPLIER
                    scored.append((ev, bt, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def _ev_digit(self, bt: BetType, _digit: int, p: float) -> float:
        """Estimate expected value for digit-based bets."""
        if bt == BetType.MOT_SO:
            return p * MOT_SO_PRIZE.get(1, 0) / 10_000
        elif bt == BetType.HAI_SO_TRUNG:
            p_at_least_2 = 3 * p * p * (1 - p) + p * p * p
            return p_at_least_2 * HAI_SO_TRUNG_PRIZE / 10_000
        elif bt == BetType.BA_SO_TRUNG:
            return p**3 * BA_SO_TRUNG_PRIZE / 10_000
        return 0.0

    def _estimate_total_prob(self, probs: dict[int, float], total: int) -> float:
        """Estimate probability of a specific total from digit probs."""
        if total <= 9:
            low = np.mean([probs.get(d, 0) for d in [1, 2, 3]])
            return low * (1 + abs(total - 10.5) * 0.05)
        elif total <= 11:
            return np.mean(list(probs.values()))
        else:
            high = np.mean([probs.get(d, 0) for d in [4, 5, 6]])
            return high * (1 + abs(total - 10.5) * 0.05)

    def _estimate_category_probs(self, probs: dict[int, float]) -> dict[str, float]:
        """Estimate category probabilities from digit probs."""
        low_prob = np.mean([probs.get(d, 0) for d in [1, 2, 3]])
        high_prob = np.mean([probs.get(d, 0) for d in [4, 5, 6]])
        p_small_raw = low_prob
        p_big_raw = high_prob
        p_draw_raw = (low_prob + high_prob) / 2
        total = p_small_raw + p_draw_raw + p_big_raw
        if total > 0:
            p_small = p_small_raw / total
            p_draw = p_draw_raw / total
            p_big = p_big_raw / total
        else:
            p_small, p_draw, p_big = 1 / 3, 1 / 3, 1 / 3
        return {"Nhỏ": p_small, "Hòa": p_draw, "Lớn": p_big}

    def _estimate_pair_prob(self, probs: dict[int, float]) -> float:
        """Estimate probability of at least 2 same digits."""
        # Approximate: P(pair) ≈ sum over digits of P(d)^2 * (1 - P(d)) * 3 + P(d)^3
        p_pair = 0.0
        for d in range(1, 7):
            p = probs.get(d, 0)
            # P(at least 2 of digit d in 3 draws)
            p_d2 = 3 * p * p * (1 - p) + p * p * p
            p_pair += p_d2
        return min(p_pair, 1.0)
