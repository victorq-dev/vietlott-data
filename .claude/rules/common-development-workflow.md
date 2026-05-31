# Development Workflow

> The full feature development process that happens before git operations.

## Feature Implementation Workflow

### 1. Plan First

- Create implementation plan before coding
- Identify dependencies and risks
- Break down into phases
- Each phase should be independently deliverable

### 2. TDD Approach

- Write tests first (RED)
- Implement to pass tests (GREEN)
- Refactor (IMPROVE)
- Verify 80%+ coverage

### 3. Code Review

- Review code immediately after writing
- Address CRITICAL and HIGH issues
- Fix MEDIUM issues when possible

### 4. Commit & Push

- Detailed commit messages
- Follow conventional commits format
- See [common-git-workflow.md](./common-git-workflow.md)

### 5. Pre-Review Checks

- Verify all automated checks (CI/CD) are passing
- Resolve any merge conflicts
- Ensure branch is up to date with target branch
