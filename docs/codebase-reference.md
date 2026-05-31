# Codebase Deep Reference

> Generated 2026-05-29. For future Claude sessions: read this file first to understand the full project without re-exploring.

## Project Overview

Python data pipeline that crawls, stores, and analyzes Vietnamese lottery (Vietlott) data. Runs daily via GitHub Actions. No server required.

**Stack**: Python 3.11+, Polars, attrs/cattrs, BeautifulSoup4, requests, Click, pendulum, loguru
**Build**: uv + setuptools, ruff lint/format, pytest
**Version**: 0.2.6 (in pyproject.toml)

---

## Directory Structure

```
src/
├── cli/                        # CLI entry points
│   ├── crawl.py                # `vietlott-crawl` command
│   └── missing.py              # `vietlott-missing` command
├── vietlott/
│   ├── config/
│   │   ├── products.py         # ProductConfig (attrs) + 7 instances + product_config_map
│   │   └── map_class.py        # map_class_name: str -> crawler class
│   ├── crawler/
│   │   ├── products/
│   │   │   ├── base.py         # BaseProduct: thread pool, crawl(), dedup, storage
│   │   │   ├── power655.py     # ProductPower655
│   │   │   ├── power645.py     # ProductPower645 (extends 655)
│   │   │   ├── power535.py     # ProductPower535
│   │   │   ├── keno.py         # ProductKeno
│   │   │   ├── p3d.py          # P3D (Max 3D)
│   │   │   ├── p3d_pro.py      # P3DPro (extends P3D)
│   │   │   └── bingo18.py      # ProductBingo18
│   │   ├── requests_helper/
│   │   │   ├── config.py       # HTTP headers + TIMEOUT=20
│   │   │   └── fetch.py        # get_vietlott_cookie(), fetch_wrapper()
│   │   ├── schema/
│   │   │   ├── requests.py     # ORenderInfoCls + per-product request classes (attrs)
│   │   │   └── response.py     # Empty (parsing done in process_result)
│   │   └── collections_helper.py # chunks_iter()
│   └── tests/
│       ├── config/             # test_bingo18, test_power535 config tests
│       ├── crawler/            # integration tests (live API)
│       └── test_crawler/       # schema + unit tests
├── render_readme.py            # `vietlott-render-readme` -> readme.md
├── render_docs.py              # `vietlott-render-docs` -> docs/index.html
└── machine_learning/
    ├── backtest.py             # StrategyBacktester, BacktestResult
    ├── render_prediction.py    # PredictionSummaryGenerator
    └── strategies/             # 9 prediction strategies
data/
├── power655.jsonl              # 1,351 records (148 KB)
├── power645.jsonl              # 1,318 records (141 KB)
├── power535.jsonl              # 595 records (63 KB)
├── keno.jsonl                  # 71,684 records (11 MB)
├── 3d.jsonl                    # 1,081 records (258 KB)
├── 3d_pro.jsonl                # 728 records (174 KB)
└── bingo18.jsonl               # 73,232 records (10 MB)
```

---

## CLI Commands (4 total)

| Command | Entry Point | Purpose |
|---------|-------------|---------|
| `vietlott-crawl PRODUCT [--run-date] [--index_from] [--index_to]` | `vietlott.cli.crawl:crawl` | Crawl lottery data for a product |
| `vietlott-missing PRODUCT [--limit N]` | `vietlott.cli.missing:detect_missing_data` | Detect + backfill missing draw IDs |
| `vietlott-render-readme` | `src.render_readme:main` | Generate readme.md with stats |
| `vietlott-render-docs` | `src.render_docs:main` | Update docs/index.html |

**Valid products**: `power_655`, `power_645`, `power_535`, `keno`, `3d`, `3d_pro`, `bingo18`

---

## Product Configs

Defined in `src/vietlott/config/products.py` as `ProductConfig` (attrs @define):

| Product | Range | Size | Interval | page_size | num_thread | default_index_to |
|---------|-------|------|----------|-----------|------------|------------------|
| power_655 | 1-55 | 6 | 2 days | 6 | 10 | 1 |
| power_645 | 1-45 | 6 | 2 days | 6 | 10 | 1 |
| power_535 | 1-35 | 5 | 2 days | 6 | 10 | 1 |
| keno | 1-45 | 6 | 8 min | 6 | 20 | 24 |
| 3d | 0-999 | 6 | 2 days | 5 | 20 | 1 |
| 3d_pro | 0-999 | 6 | 2 days | 5 | 20 | 1 |
| bingo18 | 0-9 | 3 | 5 min | 6 | 10 | 1 |

All products: `use_cookies=False`

---

## Crawler Architecture

### Inheritance Hierarchy

```
BaseProduct (base.py)
├── ProductPower655 (power655.py)
│   └── ProductPower645 (power645.py) — reuses 655's process_result
├── ProductPower535 (power535.py)
├── ProductKeno (keno.py)
├── P3D (p3d.py)
│   └── P3DPro (p3d_pro.py) — reuses P3D's process_result
└── ProductBingo18 (bingo18.py)
```

### Data Flow (BaseProduct.crawl())

1. Create `ThreadPoolExecutor(num_thread)`
2. Generate page tasks (one per page index)
3. Split into chunks via `collections_helper.chunks_iter`
4. Each thread: POST to Vietlott API → JSON → HTML in `res_json["value"]["HtmlContent"]`
5. Each subclass: `process_result()` → BeautifulSoup → list of dicts `{date, id, result, ...}`
6. Dedup against existing JSONL (set-based, O(1) per record)
7. Merge → sort by `[date, id]` → write NDJSON via Polars

### HTTP Layer

- **Endpoint**: Vietlott AjaxPro `.ashx` handlers (POST with `X-AjaxPro-Method: ServerSideDrawResult`)
- **Body**: Serialized attrs classes via `cattrs.unstructure()` → `json.dumps()`
- **Cookie**: `get_vietlott_cookie()` extracts JS-set cookie via regex (currently disabled for all products)
- **Timeout**: 20 seconds
- **No retry logic**: failures logged and skipped

### Request Schemas (attrs classes)

| Class | Product | Key Fields |
|-------|---------|------------|
| RequestPower655 | Power 6/55 | Key, GameDrawId, ArrayNumbers (5x15), PageIndex |
| RequestPower535 | Power 5/35 | Same structure as 655 |
| RequestKeno | Keno | DrawDate, GameId, OddEven, UpperLower, PageIndex |
| RequestP3D | 3D | GameId="5", CheckMulti, number01/02, PageIndex |
| RequestP3DPro | 3D Pro | GameId="7", same as P3D |
| RequestBingo18 | Bingo18 | GameId="8", DrawDate, TotalRow=45628, PageIndex |

All compose `ORenderInfoCls` (16 fields of Vietlott site metadata).

---

## Missing Data Detection & Backfill

File: `src/vietlott/cli/missing.py`

**Detection algorithm**:
1. Load JSONL → Polars DataFrame
2. Normalize IDs (strip `#`, cast to Int64)
3. Compute `diff = next_id - current_id`
4. Filter where `diff > 1` (gaps)
5. Convert gaps to page indices: `index = (last_id - id) / page_size`
6. Sort most recent first, limit to 20

**Backfill strategies**:
- Small gaps (≤50 pages): single `crawl()` call
- Large gaps (>50 pages): chunked in 20-page increments, early-stop if empty

---

## Data Format (JSONL)

Each line is a JSON object. Common fields: `date` (YYYY-MM-DD), `id` (string), `result` (varies).

| Product | result type | Extra fields |
|---------|-------------|--------------|
| Power 655/645/535 | list of int | `process_time` |
| Keno | list of 20 int | `big_small`, `odd_even` |
| Bingo18 | list of 3 int | `total`, `large_small`, `process_time` |
| 3D/3D Pro | dict: prize→list of str | — |

---

## Rendering Pipeline

### readme.md (render_readme.py)

- `ReadmeGenerator` class loads all 7 products
- Data overview: total draws, date range, total records per product
- Power 6/55 deep analysis: frequency (all-time/30d/60d/90d), days-since-last-appearance
- Output: `readme.md` at project root

### docs/index.html (render_docs.py)

- `DocsRenderer` class patches existing HTML via regex
- Updates `<tbody>` with fresh stats
- Replaces days-since section (between comment markers)
- Bilingual support (Vietnamese/English via data-vi/data-en attributes)

### ML Prediction (machine_learning/)

- 9 strategies in `strategies/` directory
- `BacktestResult` dataclass: ROI, win rate, Sharpe ratio, max drawdown
- `PredictionSummaryGenerator` → `machine_learning/readme.md`

---

## CI/CD

### GitHub Actions Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `.github/workflows/crawl.yaml` | Daily cron | Crawl data, commit, push |
| `.github/workflows/deploy-pages.yml` | Daily cron (midnight UTC) | Generate docs, deploy to GH Pages |
| `.github/workflows/publish-to-pypi.yaml` | Tag push `v*` | Build + publish to PyPI |

### Makefile Targets

| Target | Command |
|--------|---------|
| `make all` | lint + test |
| `make test` | `uv run pytest src/vietlott/tests` |
| `make lint` | ruff check + format |
| `make build` | lint + test + python -m build |
| `make pypi` | build + upload to TestPyPI |
| `make run-crawl` | crawl keno, power_535, power_655 |
| `make run-missing` | detect missing for keno, power_535 |

---

## Test Suite

**9 test functions across 5 files**. Coverage is narrow.

| File | Tests | Type |
|------|-------|------|
| tests/config/test_bingo18.py | 1 | Config validation |
| tests/config/test_power535.py | 1 | Config validation |
| tests/crawler/test_power535.py | 4 | Live API integration (655, 645, 3d, 3d_pro) |
| tests/test_crawler/test_schema.py | 1 | Schema serialization regression |
| tests/test_crawler/test_power535_schema.py | 1 | Schema structure |
| tests/test_crawler/test_fetch.py | 1 | HTML parsing unit test |

**Gaps**: No CLI tests, no error handling tests, no mocked HTTP tests, no conftest.py, no coverage measurement.

---

## Key Code Patterns

1. **Config-first design**: Adding a new product = add `ProductConfig` instance + crawler class + request schema + map entries
2. **Thread pool parallelism**: Pages fetched concurrently via `ThreadPoolExecutor`
3. **Set-based dedup**: `existing_ids = set(current_data["id"].to_list())` for O(1) lookup
4. **Polars for data**: `pl.read_ndjson()` / `pl.write_ndjson()`, lazy available via `pl.scan_ndjson()`
5. **attrs for schemas**: All request bodies are `@define` classes, serialized via `cattrs.unstructure()`
6. **Template method**: `BaseProduct.crawl()` orchestrates, subclasses only implement `process_result()`

---

## Dependencies

**Runtime**: bs4, polars, pyarrow, lxml, attrs, cattrs, click, pendulum, requests, tabulate, loguru
**Dev**: pytest, ruff, build, setuptools, wheel, twine, pre-commit
**ML (optional)**: numpy, scikit-learn, matplotlib, pandas

---

## Entry Points (pyproject.toml)

```
vietlott-crawl = "vietlott.cli.crawl:crawl"
vietlott-missing = "vietlott.cli.missing:detect_missing_data"
vietlott-render-readme = "src.render_readme:main"
vietlott-render-docs = "src.render_docs:main"
```
