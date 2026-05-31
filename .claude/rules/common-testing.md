# Testing Requirements

## Minimum Test Coverage: 80%

Test Types (ALL required):
1. **Unit Tests** - Individual functions, utilities, components
2. **Integration Tests** - API endpoints, database operations, crawler behavior

## Test-Driven Development

MANDATORY workflow:
1. Write test first (RED)
2. Run test - it should FAIL
3. Write minimal implementation (GREEN)
4. Run test - it should PASS
5. Refactor (IMPROVE)
6. Verify coverage (80%+)

## Test Structure (AAA Pattern)

Prefer Arrange-Act-Assert structure for tests:

```python
def test_merges_data_without_duplicates():
    # Arrange
    existing = [{"id": "1", "value": "a"}]
    new = [{"id": "1", "value": "a"}, {"id": "2", "value": "b"}]

    # Act
    result = merge_data(existing, new)

    # Assert
    assert len(result) == 2
    assert result[1]["id"] == "2"
```

### Test Naming

Use descriptive names that explain the behavior under test:

```python
def test_returns_empty_when_no_data_found(): ...
def test_raises_error_when_config_invalid(): ...
def test_handles_network_timeout_gracefully(): ...
```

## Troubleshooting Test Failures

1. Check test isolation
2. Verify mocks are correct
3. Fix implementation, not tests (unless tests are wrong)
