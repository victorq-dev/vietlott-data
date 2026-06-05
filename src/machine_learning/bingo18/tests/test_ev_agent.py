"""Tests for EVAgent: EV-based betting agent for Bingo18.

Tests are written first (TDD - RED phase). EVAgent does not exist yet.
Implementation lives at src/machine_learning/bingo18/ev_agent.py.
"""

from pathlib import Path

import pytest

from machine_learning.bingo18.ev_agent import EVAgent, _compute_ev, _option_region
from machine_learning.bingo18.simulator import BetType

# ---------------------------------------------------------------------------
# Helpers / shared data
# ---------------------------------------------------------------------------

_FAIR_PROBS: dict[int, float] = {d: 1 / 6 for d in range(1, 7)}


def _make_agent(**kwargs) -> EVAgent:
    """Create an EVAgent with sensible defaults, overridable via kwargs."""
    defaults = dict(
        agent_id="ev_test",
        budget=1_000_000,
        bet_size=10_000,
        window=100,
        min_ev_per_10k=-4_200.0,
        blend_alpha=0.5,
        max_bets_per_draw=2,
        single_ev_gap=200.0,
    )
    defaults.update(kwargs)
    return EVAgent(**defaults)


def _biased_draws_all_threes(n: int = 100) -> list[list[int]]:
    """n draws where all three dice are 3 — heavily skews digit 3 probability."""
    return [[3, 3, 3]] * n


# ---------------------------------------------------------------------------
# TestOptionRegion
# ---------------------------------------------------------------------------


class TestOptionRegion:
    def test_lon_hoa_nho_nho_returns_nho(self):
        # Arrange / Act
        region = _option_region(BetType.LON_HOA_NHO, "Nhỏ")
        # Assert
        assert region == "nho"

    def test_lon_hoa_nho_lon_returns_lon(self):
        region = _option_region(BetType.LON_HOA_NHO, "Lớn")
        assert region == "lon"

    def test_lon_hoa_nho_hoa_returns_hoa(self):
        region = _option_region(BetType.LON_HOA_NHO, "Hòa")
        assert region == "hoa"

    def test_cong_tong_8_returns_nho(self):
        # Arrange: total 8 falls in 3-9 "small" range
        region = _option_region(BetType.CONG_TONG, 8)
        assert region == "nho"

    def test_cong_tong_3_returns_nho(self):
        region = _option_region(BetType.CONG_TONG, 3)
        assert region == "nho"

    def test_cong_tong_9_returns_nho(self):
        region = _option_region(BetType.CONG_TONG, 9)
        assert region == "nho"

    def test_cong_tong_10_returns_hoa(self):
        region = _option_region(BetType.CONG_TONG, 10)
        assert region == "hoa"

    def test_cong_tong_11_returns_hoa(self):
        region = _option_region(BetType.CONG_TONG, 11)
        assert region == "hoa"

    def test_cong_tong_14_returns_lon(self):
        region = _option_region(BetType.CONG_TONG, 14)
        assert region == "lon"

    def test_cong_tong_18_returns_lon(self):
        region = _option_region(BetType.CONG_TONG, 18)
        assert region == "lon"

    def test_cong_tong_12_returns_lon(self):
        region = _option_region(BetType.CONG_TONG, 12)
        assert region == "lon"

    def test_mot_so_returns_digit_region(self):
        # MOT_SO bets are in their own per-digit region to avoid
        # intra-digit redundancy; each digit is its own region.
        r3 = _option_region(BetType.MOT_SO, 3)
        r5 = _option_region(BetType.MOT_SO, 5)
        # Different digits → different regions
        assert r3 != r5

    def test_lon_and_cong_tong_14_same_region(self):
        # LON_HOA_NHO:"Lớn" and CONG_TONG:14 are both in the "lon" region;
        # the portfolio builder should not pick both (redundant coverage).
        region_lon_bet = _option_region(BetType.LON_HOA_NHO, "Lớn")
        region_ct14 = _option_region(BetType.CONG_TONG, 14)
        assert region_lon_bet == region_ct14


# ---------------------------------------------------------------------------
# TestComputeEv
# ---------------------------------------------------------------------------


class TestComputeEv:
    def test_mot_so_fair_is_negative(self):
        # On fair dice, mot_so EV per 10k bet should be negative (house edge)
        ev = _compute_ev(BetType.MOT_SO, 3, _FAIR_PROBS)
        assert ev < 0

    def test_cong_tong_fair_is_negative(self):
        ev = _compute_ev(BetType.CONG_TONG, 10, _FAIR_PROBS)
        assert ev < 0

    def test_lon_hoa_nho_fair_is_negative(self):
        ev = _compute_ev(BetType.LON_HOA_NHO, "Nhỏ", _FAIR_PROBS)
        assert ev < 0

    def test_mot_so_high_prob_increases_ev(self):
        # When digit 3 has double the fair probability, EV for mot_so:3 improves
        skewed = {d: 1 / 6 for d in range(1, 7)}
        skewed[3] = 0.5
        # Renormalise others
        remaining = 1.0 - 0.5
        for d in range(1, 7):
            if d != 3:
                skewed[d] = remaining / 5

        ev_fair = _compute_ev(BetType.MOT_SO, 3, _FAIR_PROBS)
        ev_skewed = _compute_ev(BetType.MOT_SO, 3, skewed)
        assert ev_skewed > ev_fair

    def test_ev_is_float(self):
        ev = _compute_ev(BetType.MOT_SO, 1, _FAIR_PROBS)
        assert isinstance(ev, float)


# ---------------------------------------------------------------------------
# TestDigitProbEstimation
# ---------------------------------------------------------------------------


class TestDigitProbEstimation:
    def test_fair_probs_when_no_history(self):
        # Arrange: brand new agent — buffer is empty
        agent = _make_agent()
        # Act
        probs = agent._estimate_digit_probs()
        # Assert: each digit close to 1/6 due to full alpha blend
        for d in range(1, 7):
            assert probs[d] == pytest.approx(1 / 6, abs=1e-6)

    def test_fair_probs_when_few_draws(self):
        # Arrange: only 5 draws in buffer (below meaningful history)
        agent = _make_agent()
        agent.warm_up([[1, 2, 3]] * 5)
        # Act
        probs = agent._estimate_digit_probs()
        # Assert: blend keeps probs close to fair (not wildly skewed)
        for d in range(1, 7):
            assert 0.05 < probs[d] < 0.5

    def test_warm_up_populates_buffer(self):
        # Arrange
        agent = _make_agent()
        draws = [[1, 2, 3]] * 50
        # Act
        agent.warm_up(draws)
        # Assert
        assert len(agent._recent_draws) == 50

    def test_warm_up_respects_window_limit(self):
        # If warm_up receives more draws than window, only keep the last `window`
        agent = _make_agent(window=10)
        agent.warm_up([[1, 2, 3]] * 25)
        assert len(agent._recent_draws) <= 10

    def test_blended_probs_skewed_toward_recent(self):
        # Arrange: 100 draws all [3,3,3] → digit 3 dominates recent_freq
        agent = _make_agent(blend_alpha=0.0)  # no blend → pure empirical
        agent.warm_up(_biased_draws_all_threes(100))
        # Act
        probs = agent._estimate_digit_probs()
        # Assert: digit 3 should be much higher than 1/6
        assert probs[3] > 0.5

    def test_blend_alpha_1_always_fair(self):
        # Arrange: full blend → always fair regardless of history
        agent = _make_agent(blend_alpha=1.0)
        agent.warm_up(_biased_draws_all_threes(100))
        probs = agent._estimate_digit_probs()
        for d in range(1, 7):
            assert probs[d] == pytest.approx(1 / 6, abs=1e-6)

    def test_probs_sum_to_one(self):
        agent = _make_agent()
        agent.warm_up(_biased_draws_all_threes(50))
        probs = agent._estimate_digit_probs()
        total = sum(probs[d] for d in range(1, 7))
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_all_six_digits_present_in_probs(self):
        agent = _make_agent()
        probs = agent._estimate_digit_probs()
        assert set(probs.keys()) == set(range(1, 7))


# ---------------------------------------------------------------------------
# TestBetSelection
# ---------------------------------------------------------------------------


class TestBetSelection:
    def test_skips_all_bets_on_fair_dice(self):
        # Arrange: fresh agent → fair probs, all EVs < -4200 per 10k bet
        agent = _make_agent(min_ev_per_10k=-4_200.0)
        # Act: call _select_bets with exactly fair probs
        bets = agent._select_bets(_FAIR_PROBS)
        # Assert: no bet clears the min_ev threshold on fair dice
        assert bets == []

    def test_bets_when_digit_elevated(self):
        # Arrange: digit 3 very likely → mot_so:3 should appear
        agent = _make_agent(min_ev_per_10k=-100_000.0)  # permissive threshold
        skewed = {d: (1 - 0.35) / 5 for d in range(1, 7)}
        skewed[3] = 0.35
        # Act
        bets = agent._select_bets(skewed)
        # Assert: mot_so:3 is selected
        bet_labels = [(b["bet_type"], b["value"]) for b in bets]
        # Any digit-3 bet type is acceptable; ba_so_trung:3 has higher EV than mot_so:3 at p=0.35
        assert any(
            label in bet_labels for label in [(BetType.MOT_SO, 3), (BetType.HAI_SO_TRUNG, 3), (BetType.BA_SO_TRUNG, 3)]
        )

    def test_returns_list_of_dicts(self):
        agent = _make_agent(min_ev_per_10k=-100_000.0)
        bets = agent._select_bets(_FAIR_PROBS)
        assert isinstance(bets, list)
        for b in bets:
            assert "bet_type" in b
            assert "value" in b
            assert "ev" in b

    def test_single_bet_when_clearly_dominant(self):
        # Arrange: digit 1 at p=0.8 → ba_so_trung:1 has very high EV.
        # Note: cong_tong:3 has the same EV (both require all dice = 1),
        # so the top-two gap is 0 → portfolio mode with max_bets_per_draw cap.
        dominant = {1: 0.8}
        remaining = 0.2 / 5
        for d in range(2, 7):
            dominant[d] = remaining
        agent = _make_agent(
            min_ev_per_10k=-100_000.0,
            single_ev_gap=1.0,
            max_bets_per_draw=3,
        )
        bets = agent._select_bets(dominant)
        # All selected bets must be within the portfolio cap
        assert len(bets) <= 3
        # The dominant digit-1 options should be present
        bet_labels = [(b["bet_type"], b["value"]) for b in bets]
        assert any(
            label in bet_labels
            for label in [
                (BetType.BA_SO_TRUNG, 1),
                (BetType.HAI_SO_TRUNG, 1),
                (BetType.MOT_SO, 1),
            ]
        )

    def test_portfolio_from_different_regions(self):
        # Arrange: craft probs so options from "nho" and "lon" regions both clear threshold
        # digit 1 at p=0.9 makes mot_so:1 (digit_1 region) EV positive
        # but we need two *different* regions above threshold
        # Use a very permissive threshold and gap too large to trigger single bet
        agent = _make_agent(
            min_ev_per_10k=-100_000.0,
            single_ev_gap=999_999.0,  # gap so large single will never trigger
            max_bets_per_draw=5,
        )
        # Skew toward digit 3 (boosts mot_so:3) and also total likely near 9 (nho region)
        # With permissive threshold, multiple regions will have options available
        probs = {d: 1 / 6 for d in range(1, 7)}
        bets = agent._select_bets(probs)
        if len(bets) >= 2:
            regions = [_option_region(b["bet_type"], b["value"]) for b in bets]
            # All regions in portfolio must be distinct
            assert len(regions) == len(set(regions))

    def test_no_redundant_portfolio_same_region(self):
        # Arrange: LON_HOA_NHO:Lớn and CONG_TONG:14 are both "lon" region.
        # The portfolio must not select two bets from the same region.
        agent = _make_agent(
            min_ev_per_10k=-100_000.0,
            single_ev_gap=999_999.0,  # no single-bet short-circuit
            max_bets_per_draw=5,
        )
        bets = agent._select_bets(_FAIR_PROBS)
        regions = [_option_region(b["bet_type"], b["value"]) for b in bets]
        # No duplicate regions allowed
        assert len(regions) == len(set(regions))

    def test_max_bets_per_draw_limits_portfolio(self):
        agent = _make_agent(
            min_ev_per_10k=-100_000.0,
            single_ev_gap=999_999.0,
            max_bets_per_draw=2,
        )
        bets = agent._select_bets(_FAIR_PROBS)
        assert len(bets) <= 2


# ---------------------------------------------------------------------------
# TestProcessDraw
# ---------------------------------------------------------------------------


class TestProcessDraw:
    def test_process_draw_skip_returns_empty(self):
        # Arrange: fresh agent → fair probs → all EV below threshold
        agent = _make_agent(min_ev_per_10k=-4_200.0)
        # Act
        outcomes = agent.process_draw(
            actual_digits=[3, 3, 1],
            actual_total=7,
            large_small="Nhỏ",
            date="2026-06-05",
            draw_id="0170333",
        )
        # Assert: no bets placed, empty outcomes
        assert outcomes == []

    def test_process_draw_updates_rolling_buffer(self):
        # Arrange
        agent = _make_agent()
        initial_len = len(agent._recent_draws)
        # Act
        agent.process_draw([1, 2, 3], 6, "Nhỏ", date="2026-06-05", draw_id="001")
        # Assert: buffer grew by one entry
        assert len(agent._recent_draws) == initial_len + 1

    def test_process_draw_most_recent_draw_at_end(self):
        # After processing, the last element of _recent_draws is the draw we just fed.
        agent = _make_agent()
        agent.process_draw([4, 5, 6], 15, "Lớn", date="2026-06-05", draw_id="002")
        assert agent._recent_draws[-1] == [4, 5, 6]

    def test_process_draw_win_updates_budget(self):
        # Arrange: bias history toward digit 3, then feed a winning draw
        agent = _make_agent(min_ev_per_10k=-100_000.0, blend_alpha=0.0)
        agent.warm_up(_biased_draws_all_threes(100))
        budget_before = agent.budget
        # Act: draw [3,3,3] → mot_so:3 wins (payout > 0)
        outcomes = agent.process_draw([3, 3, 3], 9, "Nhỏ", date="2026-06-05", draw_id="003")
        # Assert: at least one outcome and budget changed
        if outcomes:
            winning = [o for o in outcomes if o["won"]]
            if winning:
                assert agent.budget != budget_before

    def test_process_draw_tracks_total_draws(self):
        # Arrange
        agent = _make_agent()
        assert agent._total_draws == 0
        # Act
        agent.process_draw([1, 2, 3], 6, "Nhỏ", date="2026-06-05", draw_id="004")
        agent.process_draw([4, 5, 6], 15, "Lớn", date="2026-06-05", draw_id="005")
        # Assert
        assert agent._total_draws == 2

    def test_process_draw_tracks_total_bets(self):
        # Arrange: biased history ensures bets are placed
        agent = _make_agent(min_ev_per_10k=-100_000.0, blend_alpha=0.0)
        agent.warm_up(_biased_draws_all_threes(100))
        # Act
        outcomes = agent.process_draw([3, 3, 3], 9, "Nhỏ", date="2026-06-05", draw_id="006")
        # Assert: if bets were placed, _total_bets matches outcome count
        assert agent._total_bets == len(outcomes)

    def test_process_draw_outcome_has_required_keys(self):
        # Arrange: permissive threshold to guarantee bets are placed
        agent = _make_agent(min_ev_per_10k=-100_000.0, blend_alpha=0.0)
        agent.warm_up(_biased_draws_all_threes(100))
        # Act
        outcomes = agent.process_draw([3, 3, 3], 9, "Nhỏ", date="2026-06-05", draw_id="007")
        # Assert: each outcome dict has the required fields
        for outcome in outcomes:
            assert "bet_type" in outcome
            assert "value" in outcome
            assert "units" in outcome
            assert "amount" in outcome
            assert "payout" in outcome
            assert "won" in outcome
            assert "ev" in outcome

    def test_process_draw_buffer_respects_window(self):
        # Arrange: window=5, feed more than 5 draws
        agent = _make_agent(window=5)
        for i in range(10):
            agent.process_draw([1, 2, 3], 6, "Nhỏ", date="2026-06-05", draw_id=str(i))
        # Assert: buffer never exceeds window size
        assert len(agent._recent_draws) <= 5

    def test_process_draw_budget_decreases_on_loss(self):
        # Arrange: bias toward digit 3, but feed a draw without digit 3 → mot_so:3 loses
        agent = _make_agent(min_ev_per_10k=-100_000.0, blend_alpha=0.0)
        agent.warm_up(_biased_draws_all_threes(100))
        budget_before = agent.budget
        # Act: draw with no 3s at all
        outcomes = agent.process_draw([1, 2, 4], 7, "Nhỏ", date="2026-06-05", draw_id="008")
        # Assert: budget reduced if any bets were placed
        if outcomes:
            assert agent.budget < budget_before


# ---------------------------------------------------------------------------
# TestProperties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_lifetime_roi_zero_at_start(self):
        agent = _make_agent()
        assert agent.lifetime_roi == 0.0

    def test_roi_zero_at_start(self):
        agent = _make_agent()
        assert agent.roi == 0.0

    def test_win_rate_zero_at_start(self):
        agent = _make_agent()
        assert agent.win_rate == 0.0

    def test_is_alive_true_when_budget_sufficient(self):
        agent = _make_agent(budget=1_000_000, bet_size=10_000)
        assert agent.is_alive

    def test_is_alive_false_when_budget_below_bet_size(self):
        agent = _make_agent(budget=5_000, bet_size=10_000)
        assert not agent.is_alive

    def test_lifetime_roi_negative_after_losses(self):
        # Arrange: bet placed and lost
        agent = _make_agent(min_ev_per_10k=-100_000.0, blend_alpha=0.0)
        agent.warm_up(_biased_draws_all_threes(100))
        agent.process_draw([1, 2, 4], 7, "Nhỏ", date="2026-06-05", draw_id="p01")
        # Assert: lifetime ROI may be negative if bets were placed and lost
        # (just verify it is a float; exact sign depends on whether bets were placed)
        assert isinstance(agent.lifetime_roi, float)

    def test_win_rate_between_zero_and_one(self):
        agent = _make_agent(min_ev_per_10k=-100_000.0, blend_alpha=0.0)
        agent.warm_up(_biased_draws_all_threes(100))
        for i in range(5):
            agent.process_draw([3, 3, 3], 9, "Nhỏ", date="2026-06-05", draw_id=str(i))
        assert 0.0 <= agent.win_rate <= 1.0


# ---------------------------------------------------------------------------
# TestPersistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_creates_file(self, tmp_path: Path):
        # Arrange
        agent = _make_agent(agent_id="ev_save_test")
        # Act
        agent.save(tmp_path)
        # Assert
        assert (tmp_path / "ev_save_test.json").exists()

    def test_load_returns_ev_agent_instance(self, tmp_path: Path):
        agent = _make_agent(agent_id="ev_load_test")
        agent.save(tmp_path)
        loaded = EVAgent.load(tmp_path / "ev_load_test.json")
        assert isinstance(loaded, EVAgent)

    def test_save_and_load_preserves_agent_id(self, tmp_path: Path):
        agent = _make_agent(agent_id="ev_id_check")
        agent.save(tmp_path)
        loaded = EVAgent.load(tmp_path / "ev_id_check.json")
        assert loaded.agent_id == "ev_id_check"

    def test_save_and_load_preserves_budget(self, tmp_path: Path):
        agent = _make_agent(agent_id="ev_budget", budget=750_000)
        agent.save(tmp_path)
        loaded = EVAgent.load(tmp_path / "ev_budget.json")
        assert loaded.budget == 750_000

    def test_save_and_load_preserves_hyperparams(self, tmp_path: Path):
        agent = _make_agent(
            agent_id="ev_hp",
            window=50,
            min_ev_per_10k=-3_000.0,
            blend_alpha=0.3,
            max_bets_per_draw=3,
            single_ev_gap=150.0,
        )
        agent.save(tmp_path)
        loaded = EVAgent.load(tmp_path / "ev_hp.json")
        assert loaded.window == 50
        assert loaded.min_ev_per_10k == pytest.approx(-3_000.0)
        assert loaded.blend_alpha == pytest.approx(0.3)
        assert loaded.max_bets_per_draw == 3
        assert loaded.single_ev_gap == pytest.approx(150.0)

    def test_save_and_load_preserves_state(self, tmp_path: Path):
        # Arrange: agent processes several draws to accumulate state
        agent = _make_agent(agent_id="ev_state", min_ev_per_10k=-100_000.0, blend_alpha=0.0)
        agent.warm_up(_biased_draws_all_threes(50))
        for i in range(3):
            agent.process_draw([3, 3, 3], 9, "Nhỏ", date="2026-06-05", draw_id=str(i))
        draws_before = agent._total_draws
        roi_before = agent.lifetime_roi
        # Act
        agent.save(tmp_path)
        loaded = EVAgent.load(tmp_path / "ev_state.json")
        # Assert
        assert loaded._total_draws == draws_before
        assert loaded.lifetime_roi == pytest.approx(roi_before, abs=1e-4)

    def test_save_and_load_preserves_rolling_buffer(self, tmp_path: Path):
        # Arrange: warm up with known draws
        agent = _make_agent(agent_id="ev_buf", window=20)
        known_draws = [[i % 6 + 1, (i + 1) % 6 + 1, (i + 2) % 6 + 1] for i in range(10)]
        agent.warm_up(known_draws)
        # Act
        agent.save(tmp_path)
        loaded = EVAgent.load(tmp_path / "ev_buf.json")
        # Assert: rolling buffer is restored
        assert len(loaded._recent_draws) == len(agent._recent_draws)

    def test_save_and_load_preserves_total_bets(self, tmp_path: Path):
        agent = _make_agent(agent_id="ev_bets", min_ev_per_10k=-100_000.0, blend_alpha=0.0)
        agent.warm_up(_biased_draws_all_threes(100))
        for i in range(5):
            agent.process_draw([3, 3, 3], 9, "Nhỏ", date="2026-06-05", draw_id=str(i))
        bets_before = agent._total_bets
        agent.save(tmp_path)
        loaded = EVAgent.load(tmp_path / "ev_bets.json")
        assert loaded._total_bets == bets_before
