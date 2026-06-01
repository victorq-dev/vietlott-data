"""Tests for Bingo18 simulation engine."""

import numpy as np
import pandas as pd
import pytest

from machine_learning.bingo18.model import Bingo18Model
from machine_learning.bingo18.simulator import (
    BA_SO_TRUNG_PRIZE,
    HAI_SO_TRUNG_PRIZE,
    BetType,
    Bingo18Simulator,
    SimulationResult,
    calculate_payout,
)


@pytest.fixture
def sample_df():
    """Generate synthetic Bingo18 data with digits 1-6."""
    np.random.seed(42)
    n = 500
    results = [sorted(np.random.choice(range(1, 7), 3, replace=True).tolist()) for _ in range(n)]
    totals = [sum(r) for r in results]
    large_small = ["Lớn" if t >= 12 else ("Hòa" if t >= 10 else "Nhỏ") for t in totals]
    return pd.DataFrame(
        {
            "date": [f"2025-01-{i:02d}" for i in range(1, n + 1)],
            "id": [f"{i:07d}" for i in range(1, n + 1)],
            "result": results,
            "total": totals,
            "large_small": large_small,
        }
    )


@pytest.fixture
def trained_model(sample_df):
    """Train a model on sample data."""
    model = Bingo18Model(window=30, n_estimators=10, max_depth=2)
    model.train(sample_df)
    return model


# --- Tests for calculate_payout ---


def test_mot_so_1_match():
    matches, payout = calculate_payout(BetType.MOT_SO, 5, [5, 2, 3], 10_000)
    assert matches == 1
    assert payout == 12_000


def test_mot_so_2_matches():
    matches, payout = calculate_payout(BetType.MOT_SO, 5, [5, 2, 5], 10_000)
    assert matches == 2
    assert payout == 20_000


def test_mot_so_3_matches():
    matches, payout = calculate_payout(BetType.MOT_SO, 5, [5, 5, 5], 10_000)
    assert matches == 3
    assert payout == 30_000


def test_mot_so_0_matches():
    matches, payout = calculate_payout(BetType.MOT_SO, 5, [1, 2, 3], 10_000)
    assert matches == 0
    assert payout == 0


def test_hai_so_trung_2_matches():
    matches, payout = calculate_payout(BetType.HAI_SO_TRUNG, 3, [3, 5, 3], 10_000)
    assert matches == 2
    assert payout == HAI_SO_TRUNG_PRIZE


def test_hai_so_trung_3_matches():
    matches, payout = calculate_payout(BetType.HAI_SO_TRUNG, 3, [3, 3, 3], 10_000)
    assert matches == 3
    assert payout == HAI_SO_TRUNG_PRIZE


def test_hai_so_trung_1_match():
    matches, payout = calculate_payout(BetType.HAI_SO_TRUNG, 3, [3, 1, 2], 10_000)
    assert matches == 1
    assert payout == 0


def test_ba_so_trung_exact():
    matches, payout = calculate_payout(BetType.BA_SO_TRUNG, 4, [4, 4, 4], 10_000)
    assert matches == 3
    assert payout == BA_SO_TRUNG_PRIZE


def test_ba_so_trung_not_exact():
    matches, payout = calculate_payout(BetType.BA_SO_TRUNG, 4, [4, 4, 5], 10_000)
    assert matches == 2
    assert payout == 0


def test_cong_tong_hit():
    matches, payout = calculate_payout(BetType.CONG_TONG, 12, [4, 5, 3], 10_000)
    assert matches == 1
    assert payout == 47_000


def test_cong_tong_miss():
    matches, payout = calculate_payout(BetType.CONG_TONG, 12, [1, 2, 3], 10_000)
    assert matches == 0
    assert payout == 0


def test_lon_hoa_nho_lon():
    matches, payout = calculate_payout(BetType.LON_HOA_NHO, "Lớn", [4, 5, 3], 10_000)
    assert matches == 1
    assert payout == 15_000


def test_lon_hoa_nho_nho():
    matches, payout = calculate_payout(BetType.LON_HOA_NHO, "Nhỏ", [1, 2, 3], 10_000)
    assert matches == 1
    assert payout == 15_000


def test_lon_hoa_nho_hoa():
    matches, payout = calculate_payout(BetType.LON_HOA_NHO, "Hòa", [3, 4, 3], 10_000)
    assert matches == 1
    assert payout == 20_000


def test_payout_scales_with_bet_size():
    _, payout_10k = calculate_payout(BetType.MOT_SO, 5, [5, 2, 3], 10_000)
    _, payout_20k = calculate_payout(BetType.MOT_SO, 5, [5, 2, 3], 20_000)
    assert payout_20k == payout_10k * 2


# --- Tests for Bingo18Simulator ---


def test_simulation_runs(trained_model, sample_df):
    sim = Bingo18Simulator(model=trained_model, budget=1_000_000, bet_size=10_000, bet_type="mot_so")
    result = sim.run(sample_df)

    assert isinstance(result, SimulationResult)
    assert result.starting_budget == 1_000_000
    assert result.total_bets > 0
    assert len(result.bet_history) > 0
    assert len(result.profit_curve) > 0


def test_budget_deducted_per_bet(trained_model, sample_df):
    sim = Bingo18Simulator(model=trained_model, budget=1_000_000, bet_size=10_000, bet_type="mot_so")
    result = sim.run(sample_df)

    for bet in result.bet_history:
        assert bet.bet_amount == 10_000
        assert bet.payout >= 0


def test_stops_when_broke(trained_model, sample_df):
    sim = Bingo18Simulator(model=trained_model, budget=50_000, bet_size=10_000, bet_type="mot_so")
    result = sim.run(sample_df)

    assert result.total_bets <= len(sample_df) - 30
    assert result.final_budget >= 0


def test_win_count_matches_history(trained_model, sample_df):
    sim = Bingo18Simulator(model=trained_model, budget=1_000_000, bet_size=10_000, bet_type="mot_so")
    result = sim.run(sample_df)

    wins_from_history = sum(1 for b in result.bet_history if b.payout > 0)
    assert result.wins == wins_from_history


def test_mot_so_payout_in_range(trained_model, sample_df):
    sim = Bingo18Simulator(model=trained_model, budget=10_000_000, bet_size=10_000, bet_type="mot_so")
    result = sim.run(sample_df)

    for bet in result.bet_history:
        assert bet.payout in [0, 12_000, 20_000, 30_000]


def test_cong_tong_bet_type(trained_model, sample_df):
    sim = Bingo18Simulator(
        model=trained_model,
        budget=1_000_000,
        bet_size=10_000,
        bet_type="cong_tong",
        target_total=12,
    )
    result = sim.run(sample_df)

    assert result.total_bets > 0
    assert result.bet_type == "cong_tong"
    for bet in result.bet_history:
        assert bet.bet_value == 12


def test_lon_hoa_nho_bet_type(trained_model, sample_df):
    sim = Bingo18Simulator(
        model=trained_model,
        budget=1_000_000,
        bet_size=10_000,
        bet_type="lon_hoa_nho",
        target_category="Lớn",
    )
    result = sim.run(sample_df)

    assert result.total_bets > 0
    assert result.bet_type == "lon_hoa_nho"


def test_hai_so_trung_bet_type(trained_model, sample_df):
    sim = Bingo18Simulator(model=trained_model, budget=1_000_000, bet_size=10_000, bet_type="hai_so_trung")
    result = sim.run(sample_df)

    assert result.total_bets > 0
    for bet in result.bet_history:
        # Payout should be 0 or 75,000
        assert bet.payout in [0, 75_000]


def test_ba_so_trung_bet_type(trained_model, sample_df):
    sim = Bingo18Simulator(model=trained_model, budget=10_000_000, bet_size=10_000, bet_type="ba_so_trung")
    result = sim.run(sample_df)

    assert result.total_bets > 0
    for bet in result.bet_history:
        # Payout should be 0 or 1,200,000
        assert bet.payout in [0, 1_200_000]


def test_result_to_dict(trained_model, sample_df):
    sim = Bingo18Simulator(model=trained_model, budget=1_000_000, bet_size=10_000, bet_type="mot_so")
    result = sim.run(sample_df)

    d = result.to_dict()
    assert "starting_budget" in d
    assert "final_budget" in d
    assert "profit" in d
    assert "roi_pct" in d
    assert "win_rate" in d
    assert "bet_type" in d


def test_max_drawdown_non_negative(trained_model, sample_df):
    sim = Bingo18Simulator(model=trained_model, budget=1_000_000, bet_size=10_000, bet_type="mot_so")
    result = sim.run(sample_df)

    assert result.max_drawdown >= 0


# --- Tests for new multiplier-based bet types ---


def test_cong_tong_mult_hit():
    """Multiplier-based total bet: 10k bet on total 3 should pay 10k * 120 = 1.2M."""
    matches, payout = calculate_payout(BetType.CONG_TONG_MULT, 3, [1, 1, 1], 10_000)
    assert matches == 1
    assert payout == 1_200_000


def test_cong_tong_mult_miss():
    matches, payout = calculate_payout(BetType.CONG_TONG_MULT, 3, [1, 2, 3], 10_000)
    assert matches == 0
    assert payout == 0


def test_cong_tong_mult_middle():
    """Total 10 with 10k bet = 10k * 4.4 = 44k."""
    matches, payout = calculate_payout(BetType.CONG_TONG_MULT, 10, [3, 3, 4], 10_000)
    assert matches == 1
    assert payout == 44_000


def test_lon_hoa_nho_v2_nho():
    """Nhỏ (3-9) with multiplier 1.5: 10k * 1.5 = 15k."""
    matches, payout = calculate_payout(BetType.LON_HOA_NHO_V2, "Nhỏ", [1, 2, 3], 10_000)
    assert matches == 1
    assert payout == 15_000


def test_lon_hoa_nho_v2_hoa():
    """Hòa (10-11) with multiplier 2.0: 10k * 2.0 = 20k."""
    matches, payout = calculate_payout(BetType.LON_HOA_NHO_V2, "Hòa", [3, 3, 4], 10_000)
    assert matches == 1
    assert payout == 20_000


def test_lon_hoa_nho_v2_lon():
    """Lớn (12-18) with multiplier 1.5: 10k * 1.5 = 15k."""
    matches, payout = calculate_payout(BetType.LON_HOA_NHO_V2, "Lớn", [4, 4, 5], 10_000)
    assert matches == 1
    assert payout == 15_000


def test_trung_2so_hit():
    """Specific digit pair: bet on 3, draw [3,3,5] = 10k * 7.5 = 75k."""
    matches, payout = calculate_payout(BetType.TRUNG_2SO, 3, [3, 3, 5], 10_000)
    assert matches == 2
    assert payout == 75_000


def test_trung_2so_miss():
    """Specific digit pair: bet on 4, draw [1, 2, 3] = no match."""
    matches, payout = calculate_payout(BetType.TRUNG_2SO, 4, [1, 2, 3], 10_000)
    assert matches == 0
    assert payout == 0


def test_trung_2so_wrong_digit():
    """Specific digit pair: bet on 1, draw [3, 3, 5] = digit 1 not paired."""
    matches, payout = calculate_payout(BetType.TRUNG_2SO, 1, [3, 3, 5], 10_000)
    assert matches == 0
    assert payout == 0


def test_trung_3so_specific_hit():
    """Specific triple: bet on 5, draw [5,5,5] = 10k * 120 = 1.2M."""
    matches, payout = calculate_payout(BetType.TRUNG_3SO, 5, [5, 5, 5], 10_000)
    assert matches == 3
    assert payout == 1_200_000


def test_trung_3so_specific_miss():
    matches, payout = calculate_payout(BetType.TRUNG_3SO, 5, [5, 5, 3], 10_000)
    assert matches == 2
    assert payout == 0


def test_trung_3so_any_hit():
    """Specific digit triple: bet on 4, draw [4,4,4] = 10k * 20 = 200k."""
    matches, payout = calculate_payout(BetType.TRUNG_3SO_ANY, 4, [4, 4, 4], 10_000)
    assert matches == 3
    assert payout == 200_000


def test_trung_3so_any_miss():
    """Specific digit triple: bet on 4, draw [1, 2, 3] = no match."""
    matches, payout = calculate_payout(BetType.TRUNG_3SO_ANY, 4, [1, 2, 3], 10_000)
    assert matches == 0
    assert payout == 0


def test_trung_3so_any_wrong_digit():
    """Specific digit triple: bet on 5, draw [4, 4, 4] = digit 5 not tripled."""
    matches, payout = calculate_payout(BetType.TRUNG_3SO_ANY, 5, [4, 4, 4], 10_000)
    assert matches == 0
    assert payout == 0


# --- Tests for combined simulation ---


def test_combined_simulation_runs(trained_model, sample_df):
    sim = Bingo18Simulator(model=trained_model, budget=5_000_000, bet_size=10_000)
    result = sim.run_combined(
        sample_df,
        bet_types=["mot_so", "cong_tong_mult", "lon_hoa_nho_v2"],
        mode="combine",
    )
    assert result.total_bets > 0
    assert result.starting_budget == 5_000_000


def test_all_in_simulation_runs(trained_model, sample_df):
    sim = Bingo18Simulator(model=trained_model, budget=5_000_000, bet_size=10_000)
    result = sim.run_combined(
        sample_df,
        bet_types=["mot_so", "cong_tong_mult"],
        mode="all_in",
    )
    assert result.total_bets > 0


def test_skip_simulation_skips_low_confidence(trained_model, sample_df):
    sim = Bingo18Simulator(model=trained_model, budget=5_000_000, bet_size=10_000)
    result_high = sim.run_combined(
        sample_df,
        bet_types=["mot_so"],
        mode="skip",
        confidence_threshold=0.0,
    )
    result_low = sim.run_combined(
        sample_df,
        bet_types=["mot_so"],
        mode="skip",
        confidence_threshold=0.99,
    )
    # Higher threshold should result in fewer bets
    assert result_low.total_bets <= result_high.total_bets


def test_combined_with_new_bet_types(trained_model, sample_df):
    sim = Bingo18Simulator(model=trained_model, budget=10_000_000, bet_size=10_000)
    result = sim.run_combined(
        sample_df,
        bet_types=["cong_tong_mult", "lon_hoa_nho_v2", "trung_2so", "trung_3so_any"],
        mode="combine",
    )
    assert result.total_bets > 0
    assert result.bet_type == "cong_tong_mult+lon_hoa_nho_v2+trung_2so+trung_3so_any"
