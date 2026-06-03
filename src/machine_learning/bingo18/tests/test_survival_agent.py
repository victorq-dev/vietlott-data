"""Tests for SurvivalAgent: absence-based portfolio betting."""

import json
import tempfile
from pathlib import Path

import pytest

from machine_learning.bingo18.simulator import BetType
from machine_learning.bingo18.survival_agent import (
    ALL_BET_OPTIONS,
    BetOptionState,
    SurvivalAgent,
    _expected_gap,
)

# ---------------------------------------------------------------------------
# BetOptionState
# ---------------------------------------------------------------------------


class TestBetOptionState:
    def test_absence_ratio_zero_at_start(self):
        s = BetOptionState(expected_gap=10.0)
        assert s.absence_ratio == 0.0

    def test_absence_ratio_after_draws(self):
        s = BetOptionState(expected_gap=10.0)
        s.draws_since_win = 5
        assert s.absence_ratio == pytest.approx(0.5)

    def test_absence_ratio_overdue(self):
        s = BetOptionState(expected_gap=10.0)
        s.draws_since_win = 20
        assert s.absence_ratio == pytest.approx(2.0)

    def test_roi_zero_when_no_bets(self):
        s = BetOptionState(expected_gap=5.0)
        assert s.roi == 0.0

    def test_roi_positive(self):
        s = BetOptionState(expected_gap=5.0, total_wagered=10_000, total_payout=15_000, total_bets=1)
        assert s.roi == pytest.approx(50.0)

    def test_roi_negative(self):
        s = BetOptionState(expected_gap=5.0, total_wagered=10_000, total_payout=0, total_bets=1)
        assert s.roi == pytest.approx(-100.0)

    def test_win_rate_zero_when_no_bets(self):
        s = BetOptionState(expected_gap=5.0)
        assert s.win_rate == 0.0

    def test_win_rate(self):
        s = BetOptionState(expected_gap=5.0, total_bets=4, wins=1)
        assert s.win_rate == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# _expected_gap
# ---------------------------------------------------------------------------


class TestExpectedGap:
    def test_mot_so_all_digits_equal(self):
        gaps = [_expected_gap(BetType.MOT_SO, d) for d in range(1, 7)]
        assert all(g == pytest.approx(gaps[0]) for g in gaps)

    def test_mot_so_approx(self):
        # P(digit appears) = 1 - (5/6)^3 = 91/216
        assert _expected_gap(BetType.MOT_SO, 1) == pytest.approx(216 / 91, rel=1e-3)

    def test_hai_so_trung(self):
        # P(>=2 times) = 16/216
        assert _expected_gap(BetType.HAI_SO_TRUNG, 3) == pytest.approx(216 / 16, rel=1e-3)

    def test_ba_so_trung(self):
        # P(3 times) = 1/216
        assert _expected_gap(BetType.BA_SO_TRUNG, 5) == pytest.approx(216.0, rel=1e-3)

    def test_cong_tong_most_common(self):
        # sum=10 or 11: 27/216 ways
        assert _expected_gap(BetType.CONG_TONG, 10) == pytest.approx(216 / 27, rel=1e-3)

    def test_cong_tong_rarest(self):
        # sum=3 or 18: 1/216 ways
        assert _expected_gap(BetType.CONG_TONG, 3) == pytest.approx(216.0, rel=1e-3)
        assert _expected_gap(BetType.CONG_TONG, 18) == pytest.approx(216.0, rel=1e-3)

    def test_lon_hoa_nho_nho_lon(self):
        # P(Nhỏ or Lớn) = 81/216
        assert _expected_gap(BetType.LON_HOA_NHO, "Nhỏ") == pytest.approx(216 / 81, rel=1e-3)
        assert _expected_gap(BetType.LON_HOA_NHO, "Lớn") == pytest.approx(216 / 81, rel=1e-3)

    def test_lon_hoa_nho_hoa(self):
        # P(Hòa) = 54/216
        assert _expected_gap(BetType.LON_HOA_NHO, "Hòa") == pytest.approx(216 / 54, rel=1e-3)


# ---------------------------------------------------------------------------
# ALL_BET_OPTIONS
# ---------------------------------------------------------------------------


class TestAllBetOptions:
    def test_correct_count(self):
        # 6 mot_so + 6 hai + 6 ba + 16 cong_tong + 3 lon_hoa_nho = 37
        assert len(ALL_BET_OPTIONS) == 37

    def test_contains_mot_so_all_digits(self):
        for d in range(1, 7):
            assert (BetType.MOT_SO, d) in ALL_BET_OPTIONS

    def test_contains_cong_tong_all_sums(self):
        for t in range(3, 19):
            assert (BetType.CONG_TONG, t) in ALL_BET_OPTIONS

    def test_contains_lon_hoa_nho(self):
        for cat in ["Nhỏ", "Hòa", "Lớn"]:
            assert (BetType.LON_HOA_NHO, cat) in ALL_BET_OPTIONS


# ---------------------------------------------------------------------------
# SurvivalAgent init
# ---------------------------------------------------------------------------


class TestSurvivalAgentInit:
    def test_default_creation(self):
        agent = SurvivalAgent("test_00", budget=500_000)
        assert agent.budget == 500_000
        assert agent.bet_size == 10_000
        assert agent.is_alive

    def test_all_options_tracked(self):
        agent = SurvivalAgent("test_01", budget=500_000)
        assert len(agent._states) == 37

    def test_filtered_bet_types(self):
        agent = SurvivalAgent(
            "test_02",
            budget=500_000,
            available_bet_types=[BetType.MOT_SO, BetType.LON_HOA_NHO],
        )
        types = {bt for bt, _ in agent._states}
        assert types == {BetType.MOT_SO, BetType.LON_HOA_NHO}

    def test_roi_zero_at_start(self):
        agent = SurvivalAgent("test_03", budget=500_000)
        assert agent.roi == 0.0

    def test_lifetime_roi_zero_at_start(self):
        agent = SurvivalAgent("test_04", budget=500_000)
        assert agent.lifetime_roi == 0.0

    def test_is_alive_false_below_bet_size(self):
        agent = SurvivalAgent("test_05", budget=5_000, bet_size=10_000)
        assert not agent.is_alive


# ---------------------------------------------------------------------------
# _check_hit
# ---------------------------------------------------------------------------


class TestCheckHit:
    def test_mot_so_hit(self):
        assert SurvivalAgent._check_hit(BetType.MOT_SO, 3, [1, 3, 5], 9, "Nhỏ")

    def test_mot_so_miss(self):
        assert not SurvivalAgent._check_hit(BetType.MOT_SO, 4, [1, 2, 3], 6, "Nhỏ")

    def test_hai_so_trung_hit(self):
        assert SurvivalAgent._check_hit(BetType.HAI_SO_TRUNG, 3, [3, 3, 5], 11, "Hòa")

    def test_hai_so_trung_miss_one(self):
        assert not SurvivalAgent._check_hit(BetType.HAI_SO_TRUNG, 3, [1, 3, 5], 9, "Nhỏ")

    def test_ba_so_trung_hit(self):
        assert SurvivalAgent._check_hit(BetType.BA_SO_TRUNG, 5, [5, 5, 5], 15, "Lớn")

    def test_ba_so_trung_miss_two(self):
        assert not SurvivalAgent._check_hit(BetType.BA_SO_TRUNG, 5, [5, 5, 3], 13, "Lớn")

    def test_cong_tong_hit(self):
        assert SurvivalAgent._check_hit(BetType.CONG_TONG, 10, [3, 3, 4], 10, "Hòa")

    def test_cong_tong_miss(self):
        assert not SurvivalAgent._check_hit(BetType.CONG_TONG, 10, [1, 2, 3], 6, "Nhỏ")

    def test_lon_hoa_nho_nho_hit(self):
        assert SurvivalAgent._check_hit(BetType.LON_HOA_NHO, "Nhỏ", [1, 2, 3], 6, "Nhỏ")

    def test_lon_hoa_nho_lon_miss(self):
        assert not SurvivalAgent._check_hit(BetType.LON_HOA_NHO, "Lớn", [1, 2, 3], 6, "Nhỏ")

    def test_lon_hoa_nho_hoa_hit(self):
        assert SurvivalAgent._check_hit(BetType.LON_HOA_NHO, "Hòa", [3, 3, 4], 10, "Hòa")


# ---------------------------------------------------------------------------
# _select_portfolio
# ---------------------------------------------------------------------------


class TestSelectPortfolio:
    def test_empty_when_nothing_overdue(self):
        agent = SurvivalAgent("test_p1", budget=500_000, min_absence_ratio=1.0)
        # All draws_since_win = 0 → absence_ratio = 0 < 1.0
        portfolio = agent._select_portfolio()
        assert portfolio == []

    def test_bets_overdue_option(self):
        agent = SurvivalAgent("test_p2", budget=500_000, min_absence_ratio=1.0)
        # Make lon_hoa_nho Hòa overdue: expected_gap ≈ 4, set draws_since_win = 5
        agent._states[(BetType.LON_HOA_NHO, "Hòa")].draws_since_win = 5
        portfolio = agent._select_portfolio()
        assert any(bt == BetType.LON_HOA_NHO and val == "Hòa" for bt, val, _ in portfolio)

    def test_units_proportional_to_absence(self):
        agent = SurvivalAgent("test_p3", budget=500_000, min_absence_ratio=1.0, max_units_per_option=3)
        # absence_ratio = 2x overdue → 2 units
        agent._states[(BetType.LON_HOA_NHO, "Hòa")].draws_since_win = 8  # 8/4 = 2.0
        portfolio = agent._select_portfolio()
        hoa_bets = [(bt, val, u) for bt, val, u in portfolio if bt == BetType.LON_HOA_NHO and val == "Hòa"]
        assert len(hoa_bets) == 1
        assert hoa_bets[0][2] == 2

    def test_units_capped_at_max_per_option(self):
        agent = SurvivalAgent("test_p4", budget=500_000, min_absence_ratio=1.0, max_units_per_option=2)
        # 10x overdue but max is 2
        agent._states[(BetType.BA_SO_TRUNG, 1)].draws_since_win = 2160
        portfolio = agent._select_portfolio()
        ba_bets = [(bt, val, u) for bt, val, u in portfolio if bt == BetType.BA_SO_TRUNG and val == 1]
        if ba_bets:
            assert ba_bets[0][2] <= 2

    def test_total_units_capped(self):
        agent = SurvivalAgent("test_p5", budget=500_000, min_absence_ratio=1.0, max_units_per_draw=3)
        # Make many options overdue
        for bt, val in list(agent._states)[:10]:
            agent._states[(bt, val)].draws_since_win = 999
        portfolio = agent._select_portfolio()
        total_units = sum(u for _, _, u in portfolio)
        assert total_units <= 3

    def test_no_bet_when_insufficient_budget(self):
        agent = SurvivalAgent("test_p6", budget=5_000, bet_size=10_000, min_absence_ratio=1.0)
        for state in agent._states.values():
            state.draws_since_win = 9999
        portfolio = agent._select_portfolio()
        assert portfolio == []

    def test_most_overdue_bet_first(self):
        agent = SurvivalAgent("test_p7", budget=500_000, min_absence_ratio=1.0, max_units_per_draw=1)
        agent._states[(BetType.LON_HOA_NHO, "Hòa")].draws_since_win = 8  # ratio 2.0
        agent._states[(BetType.LON_HOA_NHO, "Nhỏ")].draws_since_win = 5  # ratio 1.87
        portfolio = agent._select_portfolio()
        assert len(portfolio) == 1
        assert portfolio[0][0] == BetType.LON_HOA_NHO
        assert portfolio[0][1] == "Hòa"


# ---------------------------------------------------------------------------
# process_draw
# ---------------------------------------------------------------------------


class TestProcessDraw:
    def test_absence_resets_on_hit(self):
        agent = SurvivalAgent("test_d1", budget=500_000)
        agent._states[(BetType.LON_HOA_NHO, "Nhỏ")].draws_since_win = 10
        agent.process_draw([1, 2, 3], 6, "Nhỏ")
        assert agent._states[(BetType.LON_HOA_NHO, "Nhỏ")].draws_since_win == 0

    def test_absence_increments_on_miss(self):
        agent = SurvivalAgent("test_d2", budget=500_000)
        agent._states[(BetType.LON_HOA_NHO, "Lớn")].draws_since_win = 0
        agent.process_draw([1, 2, 3], 6, "Nhỏ")  # Lớn did NOT hit
        assert agent._states[(BetType.LON_HOA_NHO, "Lớn")].draws_since_win == 1

    def test_draw_counter_increments(self):
        agent = SurvivalAgent("test_d3", budget=500_000)
        agent.process_draw([1, 2, 3], 6, "Nhỏ")
        agent.process_draw([4, 5, 6], 15, "Lớn")
        assert agent._total_draws == 2

    def test_budget_decreases_on_bet(self):
        agent = SurvivalAgent("test_d4", budget=500_000, min_absence_ratio=1.0)
        agent._states[(BetType.LON_HOA_NHO, "Hòa")].draws_since_win = 5  # overdue
        agent.process_draw([1, 2, 3], 6, "Nhỏ")  # Hòa doesn't hit
        assert agent.budget < 500_000

    def test_budget_increases_on_win(self):
        agent = SurvivalAgent("test_d5", budget=500_000, min_absence_ratio=1.0)
        # Make lon_hoa_nho Nhỏ overdue, then it wins
        agent._states[(BetType.LON_HOA_NHO, "Nhỏ")].draws_since_win = 5
        budget_before = agent.budget
        agent.process_draw([1, 2, 3], 6, "Nhỏ")  # Nhỏ hits, payout 15k
        # We spent 10k and won 15k → net +5k
        assert agent.budget > budget_before

    def test_unbet_absence_still_updated(self):
        # Even options not bet this draw should have their absence counters updated
        agent = SurvivalAgent("test_d6", budget=500_000, min_absence_ratio=999.0)  # nothing overdue
        agent.process_draw([1, 2, 3], 6, "Nhỏ")
        # ba_so_trung digit 1 should have absence incremented
        assert agent._states[(BetType.BA_SO_TRUNG, 1)].draws_since_win == 1

    def test_returns_outcomes_list(self):
        agent = SurvivalAgent("test_d7", budget=500_000, min_absence_ratio=1.0)
        agent._states[(BetType.LON_HOA_NHO, "Nhỏ")].draws_since_win = 5
        outcomes = agent.process_draw([1, 2, 3], 6, "Nhỏ")
        assert isinstance(outcomes, list)
        assert len(outcomes) >= 1
        assert "bet_type" in outcomes[0]
        assert "units" in outcomes[0]
        assert "won" in outcomes[0]

    def test_profit_curve_updated(self):
        agent = SurvivalAgent("test_d8", budget=500_000)
        initial_len = len(agent._profit_curve)
        agent.process_draw([1, 2, 3], 6, "Nhỏ")
        assert len(agent._profit_curve) == initial_len + 1


# ---------------------------------------------------------------------------
# reset_budget
# ---------------------------------------------------------------------------


class TestResetBudget:
    def test_budget_restored(self):
        agent = SurvivalAgent("test_r1", budget=500_000)
        agent.budget = 50_000
        agent.reset_budget()
        assert agent.budget == 500_000

    def test_absence_counters_preserved(self):
        agent = SurvivalAgent("test_r2", budget=500_000)
        agent._states[(BetType.BA_SO_TRUNG, 1)].draws_since_win = 100
        agent.budget = 50_000
        agent.reset_budget()
        assert agent._states[(BetType.BA_SO_TRUNG, 1)].draws_since_win == 100

    def test_bet_stats_preserved(self):
        agent = SurvivalAgent("test_r3", budget=500_000)
        agent._states[(BetType.MOT_SO, 3)].total_bets = 50
        agent.budget = 50_000
        agent.reset_budget()
        assert agent._states[(BetType.MOT_SO, 3)].total_bets == 50

    def test_max_drawdown_preserved(self):
        agent = SurvivalAgent("test_r4", budget=500_000)
        agent._max_drawdown = 200_000
        agent.budget = 50_000
        agent.reset_budget()
        assert agent._max_drawdown == 200_000


# ---------------------------------------------------------------------------
# lifetime_roi
# ---------------------------------------------------------------------------


class TestLifetimeRoi:
    def test_zero_when_no_bets(self):
        agent = SurvivalAgent("test_lr1", budget=500_000)
        assert agent.lifetime_roi == 0.0

    def test_negative_after_loss(self):
        agent = SurvivalAgent("test_lr2", budget=500_000, min_absence_ratio=1.0)
        agent._states[(BetType.LON_HOA_NHO, "Hòa")].draws_since_win = 5
        agent.process_draw([1, 2, 3], 6, "Nhỏ")  # Hòa loses
        assert agent.lifetime_roi < 0.0

    def test_survives_bankruptcy_reset(self):
        agent = SurvivalAgent("test_lr3", budget=500_000, min_absence_ratio=1.0)
        # Manually set some stats
        agent._states[(BetType.MOT_SO, 1)].total_wagered = 100_000
        agent._states[(BetType.MOT_SO, 1)].total_payout = 70_000
        agent.budget = 10_000
        agent.reset_budget()
        # Lifetime ROI uses bet_type_stats, not snapshot budget
        assert agent.lifetime_roi == pytest.approx(-30.0)


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_roundtrip_basic(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = SurvivalAgent("sv_000", budget=300_000, bet_size=10_000)
            agent._states[(BetType.LON_HOA_NHO, "Hòa")].draws_since_win = 42
            agent._total_draws = 100
            agent.save(Path(tmp))

            loaded = SurvivalAgent.load(Path(tmp) / "sv_000.json")
            assert loaded.agent_id == "sv_000"
            assert loaded.budget == 300_000
            assert loaded._total_draws == 100
            assert loaded._states[(BetType.LON_HOA_NHO, "Hòa")].draws_since_win == 42

    def test_roundtrip_filtered_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = SurvivalAgent(
                "sv_001",
                budget=500_000,
                available_bet_types=[BetType.MOT_SO, BetType.CONG_TONG],
            )
            agent.save(Path(tmp))
            loaded = SurvivalAgent.load(Path(tmp) / "sv_001.json")
            loaded_types = {bt for bt, _ in loaded._states}
            assert loaded_types == {BetType.MOT_SO, BetType.CONG_TONG}

    def test_roundtrip_option_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = SurvivalAgent("sv_002", budget=500_000)
            s = agent._states[(BetType.CONG_TONG, 10)]
            s.total_bets = 5
            s.total_wagered = 50_000
            s.total_payout = 30_000
            s.wins = 1
            agent.save(Path(tmp))

            loaded = SurvivalAgent.load(Path(tmp) / "sv_002.json")
            ls = loaded._states[(BetType.CONG_TONG, 10)]
            assert ls.total_bets == 5
            assert ls.total_wagered == 50_000
            assert ls.total_payout == 30_000
            assert ls.wins == 1

    def test_roundtrip_hyperparams(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = SurvivalAgent(
                "sv_003",
                budget=500_000,
                max_units_per_draw=7,
                max_units_per_option=4,
                min_absence_ratio=1.5,
            )
            agent.save(Path(tmp))
            loaded = SurvivalAgent.load(Path(tmp) / "sv_003.json")
            assert loaded.max_units_per_draw == 7
            assert loaded.max_units_per_option == 4
            assert loaded.min_absence_ratio == pytest.approx(1.5)
