# Python Patterns

> Extends common/patterns.md with Python-specific content.

## Protocol (Duck Typing)

```python
from typing import Protocol

class Repository(Protocol):
    def find_by_id(self, id: str) -> dict | None: ...
    def save(self, entity: dict) -> dict: ...
```

## Dataclasses / Attrs as DTOs

This project uses `attrs` for schemas:

```python
from attrs import define

@define
class CreateUserRequest:
    name: str
    email: str
    age: int | None = None
```

## Context Managers & Generators

- Use context managers (`with` statement) for resource management
- Use generators for lazy evaluation and memory-efficient iteration

## Comprehensions

Prefer list/dict/set comprehensions over C-style loops:

```python
# GOOD
results = [process(item) for item in items if item.active]

# AVOID
results = []
for item in items:
    if item.active:
        results.append(process(item))
```

## Error Handling

```python
# GOOD: Specific exceptions, logged context
try:
    result = fetch_data(url)
except requests.RequestException as e:
    logger.error(f"Failed to fetch {url}: {e}")
    raise
```

- Catch specific exceptions, never bare `except:`
- Log errors with context using `loguru`
- Re-raise or wrap, don't silently swallow

## Polars Patterns

This project uses Polars for dataframes:

```python
import polars as pl

# Read data
df = pl.read_ndjson("data/keno.jsonl")

# Filter and transform
result = (
    df
    .filter(pl.col("date") > "2024-01-01")
    .sort("date", descending=True)
    .head(100)
)
```
