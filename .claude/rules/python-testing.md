# Python Testing

> Extends common/testing.md with Python-specific content.

## Framework

Use **pytest** as the testing framework.

## Running Tests

```bash
# All tests
uv run pytest src/vietlott/tests

# Single test
uv run pytest path/to/test.py::test_function

# With coverage
uv run pytest --cov=src --cov-report=term-missing
```

## Test Organization

Tests live in `src/vietlott/tests/`. Use `pytest.mark` for categorization:

```python
import pytest

@pytest.mark.unit
def test_calculate_total():
    ...

@pytest.mark.integration
def test_database_connection():
    ...
```

## Test Naming

Use descriptive names that explain the behavior under test:

```python
def test_returns_empty_when_no_results_match():
    ...

def test_raises_error_when_config_missing():
    ...

def test_merges_new_data_without_duplicates():
    ...
```

## Fixtures

Use pytest fixtures for shared setup:

```python
import pytest

@pytest.fixture
def sample_config():
    return ProductConfig(name="test", raw_path="data/test.jsonl", ...)
```

## Mocking

Mock external dependencies (HTTP requests, file I/O) but NOT internal logic:

```python
from unittest.mock import patch

@patch("vietlott.crawler.requests_helper.fetch.fetch_wrapper")
def test_crawl_handles_network_error(mock_fetch):
    mock_fetch.side_effect = ConnectionError
    ...
```
