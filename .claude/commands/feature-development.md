---
name: feature-development
description: Standard feature implementation workflow for vietlott-data.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /feature-development

Standard feature implementation workflow for the vietlott-data project.

## Goal

Implement a new feature following the project's conventions and TDD approach.

## Suggested Sequence

1. **Plan** -- Understand requirements, identify affected files, create implementation plan
2. **Test** -- Write failing tests first (RED)
3. **Implement** -- Write minimal code to pass tests (GREEN)
4. **Refactor** -- Clean up while keeping tests green (IMPROVE)
5. **Review** -- Run ruff, verify tests pass, check coverage
6. **Commit** -- Conventional commit message

## Verification

```bash
uv run ruff check --select I --fix ./src && uv run ruff format ./src
uv run pytest src/vietlott/tests
```

## Project Conventions

- All code in `/src` directory
- Use `attrs` for data structures, `polars` for dataframes
- Use `loguru` for logging, `click` for CLI
- Use `pendulum` for dates, `pathlib` for paths
- NDJSON for data storage
- Type hints required on all functions
