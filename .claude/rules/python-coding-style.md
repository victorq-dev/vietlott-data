# Python Coding Style

> Extends common/coding-style.md with Python-specific content.

## Standards

- Follow **PEP 8** conventions
- Use **type annotations** on all function signatures
- **Line length**: 120 characters (enforced by ruff)

## Formatting & Linting

- **ruff** for linting and formatting (replaces black, isort, flake8)
- Run: `uv run ruff check --select I --fix ./src && uv run ruff format ./src`

## Immutability

Prefer immutable data structures:

```python
from attrs import define

@define(frozen=True)
class User:
    name: str
    email: str

from typing import NamedTuple

class Point(NamedTuple):
    x: float
    y: float
```

## Project-Specific Conventions

- **Data structures**: Use `attrs`/`cattrs` for schemas, `polars` for dataframes
- **Data format**: NDJSON for data storage (`pl.read_ndjson()`/`pl.write_ndjson()`)
- **Logging**: Use `loguru` logger (not `print()`)
- **Paths**: Use `pathlib` for file operations
- **Dates**: Use `pendulum` for date/time handling
- **CLI**: Use `click` for command-line interfaces
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Docstrings**: Required for public functions and classes
- **Comments**: Avoid unless absolutely necessary for complex logic

## Code Smells to Avoid

### Mutable Default Arguments

```python
# BAD
def f(x=[]):
    x.append(1)
    return x

# GOOD
def f(x=None):
    if x is None:
        x = []
    x.append(1)
    return x
```

### Deep Nesting

Prefer early returns over nested conditionals.

### Magic Numbers

Use named constants or config values for meaningful thresholds.

### print() Statements

Use `loguru.logger` instead of `print()` for all output.
