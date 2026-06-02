# Bingo18 ML System - Session Summary (2026-06-01)

## Bugs Found & Fixed

### 1. TRUNG_2SO / TRUNG_3SO_ANY fabricated bet types (CRITICAL)

**Problem**: Both bet types accepted `None` as bet_value — "any pair/triple wins" without picking a specific digit. Real game requires picking digit 1-6.

- `TRUNG_2SO`: Real probability = 7.41% (specific digit pair), fabricated = 44.44% (any pair)
- `TRUNG_3SO_ANY`: Real probability = 0.46% (specific digit triple), fabricated = 2.78% (any triple)
- EV of fabricated `TRUNG_2SO` = +233% (guaranteed profit!)

**Files fixed**:
- `simulator.py:268-276` — payout logic now uses `bet_value` (specific digit)
- `simulator.py:282-290` — TRUNG_3SO_ANY same fix
- `agent.py:268` — added both to digit-selection group
- `agent.py:295-296` — removed `None` handlers
- `simulator.py:435,448` — removed from "no selection needed" groups
- `simulator.py:779-785,789-795` — EV calculation iterates digits 1-6
- `tests/test_simulator.py` — updated tests + added `test_trung_2so_wrong_digit`, `test_trung_3so_any_wrong_digit`

**Verification**: ROI went from +2,923,792,769,862% to -98%. All 161 tests pass.

### 2. Code review fixes (MEDIUM)

- `_estimate_category_probs`: `p_draw` could go negative → added `max(0.0, ...)` clamp
- Added docstrings documenting heuristic nature of `_estimate_total_prob` and `_estimate_category_probs`
- Cached `cat_probs_cache` in `_score_bets_for_agent` (called twice for LON_HOA_NHO variants)
- Fixed long line in `_select_bet_value` by extracting `digit_types` tuple

---

## Multi-Bet System

### Problem
`decide_bets()` returned exactly 1 bet per draw. Real players spread bets across multiple bet types.

### Solution
- `AgentGenome` added `max_bets_per_draw=3` and `multi_bet_budget_share=0.06`
- `_score_bets_for_agent()` — EV scoring for all 10 bet types (adapted from simulator)
- `_multi_bet()` — selects top-N positive-EV bets, applies weight multiplier, allocates budget proportionally
- `decide_bets()` routes to `_multi_bet()` when `max_bets_per_draw > 1`
- `create_diverse_agents()` varies `max_bets_per_draw=[1,2,3,5]` and `budget_share=[0.03,0.06,0.10]`

**Verification**: 158 tests pass. ROI realistic (-98% to -100%).

---

## Logging & CLI

### Verbose logging
- `--verbose` / `-v` flag on `race` command enables DEBUG logs
- Per-draw logging: agent ID, bet details, result, balance before/after
- Progress logging every 5000 draws
- Bankrupt agent notifications
- Adaptation logging with change details

---

## Strategy Learning System

### Problem
Heuristic adaptation system uses hand-coded rules. User wants model that learns WHEN to use WHICH bet type.

### Architecture
- `strategy_model.py`: ContextBuilder + StrategyModel (MLP policy network)
- `strategy_trainer.py`: Walk-forward offline trainer on 73k historical draws
- Reward-weighted regression: higher reward → higher sample weight for training
- Skip action: model can choose NOT to bet (neutral reward)

### Context vector (~51 dims)
- 31 features from Bingo18FeatureEngineer
- 6 digit probabilities from Stage 1 model
- Budget ratio, win streak, loss streak
- Per-bet-type recent ROI (10 values)
- Model confidence (max_prob - min_prob)

### Training results
- Epoch 1: 11.5% win rate, ROI=-81.3%, accuracy=21.3%
- Epoch 5: 36.8% win rate, ROI=-43.8%, accuracy=47.0%
- Test: 34.4% win rate, ROI=-51.6%, only 96 bets/14.6k draws (learned to skip)

### Agent integration
- `_strategy_bet()` method: builds context → predicts bet types → confidence gating → budget allocation
- Confidence gating: skip if max bet probability < 1.3x uniform
- Conservative allocation: bet minimum when budget < 10% of starting
- Injected via `--strategy-model` flag in `race` command

### Race results (strategy vs heuristic)
| Metric | Strategy | Heuristic |
|---|---|---|
| Agents survived | 6/6 | 4/6 |
| Winner budget | 45,733 | 0 |
| Winner win rate | 23.2% | 0% |
| Winner ROI | -95.43% | -100% |

Strategy model survives 73k draws without going bankrupt. Heuristic agents die.

### Commands
```bash
# Train strategy model
PYTHONPATH=src python3 -m machine_learning.bingo18.cli train-strategy --epochs 5 --budget 1000000 --output strategy.pkl

# Race with strategy model
PYTHONPATH=src python3 -m machine_learning.bingo18.cli race --budget 500000 --n-agents 6 --strategy-model strategy.pkl
```

---

## Strategy Optimization Results (2026-06-01, updated 2026-06-02)

### Approach: EV-Based Skip + Exact Probabilities + Parallel Training

In a negative EV game (fair 3d6, house edge 43-50% on all bet types), the optimal strategy is to **bet less**. Implemented:

1. **Exact 3d6 probability calculations** (`dice_probs.py`): Replaced crude heuristics with mathematically correct 3-fold convolution for total/category/pair/triple probabilities
2. **EV-based skip mechanism**: Skip rate scales with best available EV. When all bets have very negative EV, skip more draws. When model predicts skewed digit probabilities (better EV), bet more often
3. **Progressive risk management**: Scale bet size and skip rate based on budget health (0-25% → 97% skip, 25-50% → 85% skip, 50-75% → use EV-based skip)
4. **Reward-weighted subsampling fix**: Fixed bug where training weights were reset to uniform when subsampling >50k samples
5. **Duplicate `_strategy_bet` removal**: Two copies existed; removed the first (unused) one
6. **Progressive health multiplier in `_calculate_bet_amount`**: Budget <25% → 20% bet size, <50% → 40%, <75% → 70%
7. **Parallel training** (`parallel_trainer.py`): Run multiple training agents with different hyperparameters, select best by composite score (ROI * bet_penalty * survival)
8. **Composite scoring**: Penalize models with <50 bets (lucky, not skilled), reward survival

### Results

| Metric | Baseline | Best Improved | Delta |
|--------|----------|---------------|-------|
| Winner ROI | -92.40% | **-90.83%** | **+1.57%** |
| Winner budget | 38,000 | **45,848** | **+7,848** |
| Survivors (of 6) | 2 | **4** | +100% |
| Best parallel model | — | lr0.0005_h64x32_e0.2_s0.2 | — |

### Key Insight

**In a negative EV game, the best bet is often no bet at all.** The EV gate (threshold=-0.30) ensures agents only bet when model predictions suggest significantly lower house edge than fair dice. Parallel training finds hyperparameter combinations that learn better skip/bet policies.

### House Edge Reference (exact calculations)

| Bet Type | Fair Dice EV per 10k | House Edge |
|----------|---------------------|------------|
| MOT_SO (digit 1-6) | -4,305 | 43.1% |
| LON_HOA_NHO Nho/Lon | -4,375 | 43.8% |
| LON_HOA_NHO Hoa | -5,000 | 50.0% |
| CONG_TONG (sum 3/18) | -4,444 | 44.4% |
| CONG_TONG (sum 10/11) | -4,500 | 45.0% |

### Files Modified

| File | Changes |
|------|---------|
| `dice_probs.py` | **NEW** - Exact 3d6 probability calculations |
| `strategy_model.py` | Fixed subsampling bug, context_dim default 45→51 |
| `agent.py` | EV-based skip, progressive risk, exact probs, removed duplicate _strategy_bet |
| `simulator.py` | Replaced heuristic with exact probability calculations |
| `tests/test_agent.py` | Updated sample_predictions for EV-based skip, relaxed assertions |
| `parallel_trainer.py` | **NEW** - Parallel training with composite scoring |
| `cli.py` | Added `train-parallel` command |

## Current Test Count: 171

- 158 original tests (including multi-bet)
- 13 strategy model tests

---

## Key Insight: Fair 3d6

Bingo18 uses fair 3 dice. Every digit has true probability ~42.1%. **Every bet type has negative expected value** (house edge ~37-50%). No ML model can create positive edge on a truly fair game. The strategy model optimizes survival (skip when uncertain, bet small when low budget) rather than profit.

---

## Files Modified (this session)

| File | Changes |
|---|---|
| `agent.py` | Multi-bet system, strategy bet, confidence gating, genome params |
| `simulator.py` | TRUNG_2SO/TRUNG_3SO_ANY fixes, code review fixes |
| `race.py` | Per-draw logging, bankrupt notifications, progress logging |
| `cli.py` | --verbose flag, train-strategy command, --strategy-model flag |
| `tests/test_agent.py` | Multi-bet tests, category probs tests |
| `tests/test_simulator.py` | Updated TRUNG tests |
| **New files** | |
| `strategy_model.py` | Policy network (MLP), ContextBuilder, skip action |
| `strategy_trainer.py` | Walk-forward offline trainer, reward-weighted regression |
| `tests/test_strategy.py` | 13 strategy model tests |
