"""Exact probability calculations for 3 fair dice (Bingo18).

Provides precomputed lookup tables for fast EV scoring in agent decisions.
Replaces heuristic estimations with mathematically correct probabilities.
"""

import numpy as np
from functools import lru_cache

# Precomputed: probability of each total (3-18) for fair 3d6
# P(total=T) = number of ways to get T / 216
_TOTAL_COUNTS = {
    3: 1, 4: 3, 5: 6, 6: 10, 7: 15, 8: 21, 9: 25, 10: 27,
    11: 27, 12: 25, 13: 21, 14: 15, 15: 10, 16: 6, 17: 3, 18: 1,
}
FAIR_TOTAL_PROBS = {t: c / 216.0 for t, c in _TOTAL_COUNTS.items()}

# Precomputed: category probabilities for fair 3d6
FAIR_CATEGORY_PROBS = {
    "Nhỏ": sum(FAIR_TOTAL_PROBS[t] for t in range(3, 10)),   # 3-9
    "Hòa": sum(FAIR_TOTAL_PROBS[t] for t in [10, 11]),        # 10-11
    "Lớn": sum(FAIR_TOTAL_PROBS[t] for t in range(12, 19)),  # 12-18
}

# Precomputed: P(at least k of digit d in 3 fair dice)
# For fair dice, P(digit=d) = 1/6 for each die
_FAIR_P = 1.0 / 6.0
FAIR_P_AT_LEAST_1 = 1 - (5.0 / 6.0) ** 3  # ~0.4213
FAIR_P_AT_LEAST_2 = 3 * _FAIR_P ** 2 * (1 - _FAIR_P) + _FAIR_P ** 3  # ~0.0278
FAIR_P_EXACT_3 = _FAIR_P ** 3  # ~0.00463


def compute_total_probs(digit_probs: dict[int, float]) -> dict[int, float]:
    """Compute exact P(total=T) given non-uniform digit probabilities.

    Uses 3-fold convolution of the digit distribution.
    For fair dice, this matches FAIR_TOTAL_PROBS exactly.

    Parameters
    ----------
    digit_probs : dict mapping digit (1-6) to probability

    Returns
    -------
    dict mapping total (3-18) to probability
    """
    p = [digit_probs.get(d, 1.0 / 6.0) for d in range(1, 7)]

    # 2-fold convolution (sum of 2 dice)
    p2 = {}
    for i in range(6):
        for j in range(6):
            s = i + j + 2  # +2 because dice are 1-indexed
            p2[s] = p2.get(s, 0.0) + p[i] * p[j]

    # 3-fold convolution (sum of 3 dice)
    p3 = {}
    for s2, prob2 in p2.items():
        for k in range(6):
            s = s2 + k + 1  # k+1 is the third die value
            p3[s] = p3.get(s, 0.0) + prob2 * p[k]

    return p3


def compute_category_probs(digit_probs: dict[int, float]) -> dict[str, float]:
    """Compute exact P(Nho/Hoa/Lon) given non-uniform digit probabilities.

    Parameters
    ----------
    digit_probs : dict mapping digit (1-6) to probability

    Returns
    -------
    dict with keys "Nhỏ", "Hòa", "Lớn"
    """
    total_probs = compute_total_probs(digit_probs)
    p_small = sum(total_probs.get(t, 0.0) for t in range(3, 10))
    p_draw = sum(total_probs.get(t, 0.0) for t in [10, 11])
    p_big = sum(total_probs.get(t, 0.0) for t in range(12, 19))
    return {"Nhỏ": p_small, "Hòa": p_draw, "Lớn": p_big}


def compute_pair_prob(digit_probs: dict[int, float], digit: int) -> float:
    """Compute P(at least 2 of specific digit in 3 dice) given digit probs.

    Parameters
    ----------
    digit_probs : dict mapping digit (1-6) to probability
    digit : the specific digit to check

    Returns
    -------
    float probability
    """
    p = digit_probs.get(digit, 1.0 / 6.0)
    # P(exactly 2) = C(3,2) * p^2 * (1-p)
    # P(exactly 3) = p^3
    # P(at least 2) = 3*p^2*(1-p) + p^3
    return 3 * p * p * (1 - p) + p ** 3


def compute_triple_prob(digit_probs: dict[int, float], digit: int) -> float:
    """Compute P(all 3 dice are specific digit) given digit probs.

    Parameters
    ----------
    digit_probs : dict mapping digit (1-6) to probability
    digit : the specific digit to check

    Returns
    -------
    float probability
    """
    p = digit_probs.get(digit, 1.0 / 6.0)
    return p ** 3


def compute_mot_so_ev(digit_probs: dict[int, float], digit: int) -> float:
    """Compute exact EV for MOT_SO bet (per 10k bet).

    EV = P(1 match)*12000 + P(2 matches)*20000 + P(3 matches)*30000 - 10000

    Parameters
    ----------
    digit_probs : dict mapping digit (1-6) to probability
    digit : the digit to bet on

    Returns
    -------
    float expected value (can be negative)
    """
    p = digit_probs.get(digit, 1.0 / 6.0)
    q = 1 - p

    # P(exactly k matches in 3 dice) = C(3,k) * p^k * q^(3-k)
    p0 = q ** 3
    p1 = 3 * p * q ** 2
    p2 = 3 * p ** 2 * q
    p3 = p ** 3

    expected_payout = p1 * 12_000 + p2 * 20_000 + p3 * 30_000
    return expected_payout - 10_000


def compute_cong_tong_ev(digit_probs: dict[int, float], total: int, multiplier: bool = False) -> float:
    """Compute exact EV for CONG_TONG bet.

    Parameters
    ----------
    digit_probs : dict mapping digit (1-6) to probability
    total : the sum to bet on (3-18)
    multiplier : if True, use multiplier payout (bet_size * mult), else fixed prize

    Returns
    -------
    float expected value per 10k bet
    """
    from machine_learning.bingo18.simulator import CONG_TONG_PRIZE

    total_probs = compute_total_probs(digit_probs)
    p = total_probs.get(total, 0.0)
    prize = CONG_TONG_PRIZE.get(total, 0)
    return p * prize - 10_000


def compute_lon_hoa_nho_ev(digit_probs: dict[int, float], category: str, multiplier: bool = False) -> float:
    """Compute exact EV for LON_HOA_NHO bet.

    Parameters
    ----------
    digit_probs : dict mapping digit (1-6) to probability
    category : "Nhỏ", "Hòa", or "Lớn"
    multiplier : if True, use multiplier payout

    Returns
    -------
    float expected value per 10k bet
    """
    from machine_learning.bingo18.simulator import LON_HOA_NHO_PRIZE

    cat_probs = compute_category_probs(digit_probs)
    p = cat_probs.get(category, 0.0)
    prize = LON_HOA_NHO_PRIZE.get(category, 0)
    return p * prize - 10_000
