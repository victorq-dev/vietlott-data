# Code Review Standards

## Purpose

Code review ensures quality, security, and maintainability before code is merged.

## When to Review

**MANDATORY review triggers:**
- After writing or modifying code
- Before any commit to shared branches
- When security-sensitive code is changed
- When architectural changes are made

## Review Checklist

Before marking code complete:
- [ ] Code is readable and well-named
- [ ] Functions are focused (<50 lines)
- [ ] Files are cohesive (<800 lines)
- [ ] No deep nesting (>4 levels)
- [ ] Errors are handled explicitly
- [ ] No hardcoded secrets or credentials
- [ ] Tests exist for new functionality
- [ ] Test coverage meets 80% minimum

## Review Severity Levels

| Level | Meaning | Action |
|-------|---------|--------|
| CRITICAL | Security vulnerability or data loss risk | **BLOCK** - Must fix before merge |
| HIGH | Bug or significant quality issue | **WARN** - Should fix before merge |
| MEDIUM | Maintainability concern | **INFO** - Consider fixing |
| LOW | Style or minor suggestion | **NOTE** - Optional |

## Common Issues to Catch

### Security
- Hardcoded credentials
- Unsafe deserialization
- Path traversal
- Command injection

### Code Quality
- Large functions (>50 lines) - split into smaller
- Large files (>800 lines) - extract modules
- Deep nesting (>4 levels) - use early returns
- Missing error handling - handle explicitly
- Mutation patterns - prefer immutable operations
- Missing tests - add test coverage

### Performance
- Inefficient algorithms
- Missing caching for expensive operations
- Unbounded queries or iterations
