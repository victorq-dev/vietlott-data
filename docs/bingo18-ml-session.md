# Bingo18 ML Module — Session Summary

> Created: 2026-05-31
> Purpose: Document everything built in this session for future reference.

## Overview

Built a complete ML training + auto-play simulation system for **Bingo18** lottery product. The system trains ML models on historical data, simulates betting with a budget, and includes an auto-tuner that tries multiple algorithms to find the best strategy.

## Architecture

```
src/machine_learning/bingo18/
├── __init__.py
├── features.py          # Feature engineering (28 features)
├── model.py             # Multi-algorithm ML model
├── simulator.py         # Auto-play simulation engine (5 bet types)
├── auto_tuner.py        # Auto-tuner: find best algorithm + strategy
├── cli.py               # CLI: vietlott-bingo18-play
├── report.py            # Markdown report generator
└── tests/
    ├── __init__.py
    ├── test_features.py     # 9 tests
    ├── test_model.py        # 7 tests (incl. multi-algorithm)
    ├── test_simulator.py    # 26 tests
    └── test_auto_tuner.py   # 9 tests
```

**Total: 51 tests, all passing.**

## Key Components

### 1. Feature Engineering (`features.py`)

`Bingo18FeatureEngineer` extracts 28 features from a rolling window of draws:

- `freq_{1-6}`: Rolling frequency of each digit (normalized)
- `gap_{1-6}`: Draws since last appearance of each digit
- `sum_mean`, `sum_std`: Mean/std of totals in window
- `last_draw_{0-8}`: Flattened last 3 draws (9 features)
- `odd_ratio`, `even_ratio`: Odd/even ratio in last draw
- `big_ratio`: Ratio of "Lớn" outcomes in window
- `streak_big`, `streak_small`: Current streak length

**Important**: Bingo18 uses digits **1-6** (not 0-9 as the config suggests).

### 2. ML Model (`model.py`)

`Bingo18Model` supports **4 algorithms**:

| Algorithm | Class |
|-----------|-------|
| `gradient_boosting` | GradientBoostingClassifier |
| `random_forest` | RandomForestClassifier |
| `extra_trees` | ExtraTreesClassifier |
| `logistic_regression` | LogisticRegression |

Each algorithm trains **6 binary classifiers** (one per digit 1-6), predicting P(digit appears in next draw).

Key methods:
- `train(df, test_ratio=0.2)` → TrainingMetrics
- `predict_proba(X)` → dict[digit, probability]
- `predict_top_n(X, n)` → list[int]
- `save(path)` / `load(path)` — joblib serialization

### 3. Simulation Engine (`simulator.py`)

`Bingo18Simulator` simulates auto-play with all **5 official Bingo18 bet types**:

| Bet Type | Description | Prize |
|----------|-------------|-------|
| `mot_so` | Pick 1 number, win per match | 12k/20k/30k for 1/2/3 matches |
| `hai_so_trung` | Number appears 2+ times | 75,000 VND |
| `ba_so_trung` | All 3 digits match | 1,200,000 VND |
| `cong_tong` | Pick total sum (3-18) | 44k - 1.2M depending on total |
| `lon_hoa_nho` | Pick Big/Draw/Small | 15k/20k/15k |

Prize tables are based on `docs/rules/rule-bingo18.md`.

Betting strategies: `top_n`, `threshold`, `kelly`.

### 4. Auto-Tuner (`auto_tuner.py`)

`Bingo18AutoTuner` tries all combinations of:
- 4 algorithms
- Multiple windows (10, 30, 50)
- Multiple hyperparameters (n_estimators, max_depth)
- 5 bet types
- Multiple strategies

Runs with **multiple budget levels** (1M, 5M, 10M, 50M VND) and ranks by final_budget + bets_survived.

Can save top K models with metadata to `models/bingo18/`.

### 5. CLI (`cli.py`)

Entry point: `vietlott-bingo18-play` (registered in pyproject.toml)

```bash
# Single simulation
vietlott-bingo18-play --budget 10000000 --bet-type mot_so --algorithm gradient_boosting

# Auto-tune
vietlott-bingo18-play --budget 10000000 --auto-tune

# Auto-tune + save best
vietlott-bingo18-play --budget 10000000 --auto-tune --save-best

# Use saved model
vietlott-bingo18-play --budget 10000000 --model-path models/bingo18/best.joblib
```

Run with: `PYTHONPATH=src .venv/bin/python -m machine_learning.bingo18.cli ...`

## Data

- **File**: `data/bingo18.jsonl` — 73,251 records (NDJSON format)
- **Fields**: date, id, result (list of 3 ints), total, large_small, process_time
- **Digit range**: 1-6 (NOT 0-9)
- **Draw interval**: Every 5-6 minutes

## Dependencies Added

- numpy, scikit-learn, pandas, matplotlib, joblib (already in pyproject.toml as optional [ml] extras)

## Reality Check

All simulations show **negative ROI** (-99% to -100%). This is expected — lottery has a house edge by design. The auto-tuner finds the combination that **survives longest** before going broke, but no strategy can be profitable long-term.

## Files Modified

| File | Change |
|------|--------|
| `pyproject.toml` | Added `vietlott-bingo18-play` entry point |

## Quick Reference

```bash
# Run tests
PYTHONPATH=src .venv/bin/python -m pytest src/machine_learning/bingo18/tests -v

# Lint
.venv/bin/ruff check --fix src/machine_learning/bingo18/ && .venv/bin/ruff format src/machine_learning/bingo18/

# Train model
PYTHONPATH=src .venv/bin/python -m machine_learning.bingo18.cli --budget 10000000 --train-only --model-path model.joblib

# Run simulation
PYTHONPATH=src .venv/bin/python -m machine_learning/bingo18.cli --budget 10000000 --bet-type mot_so --model-path model.joblib
```
