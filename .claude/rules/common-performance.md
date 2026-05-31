# Performance Optimization

## Context Window Management

Avoid last 20% of context window for:
- Large-scale refactoring
- Feature implementation spanning multiple files
- Debugging complex interactions

Lower context sensitivity tasks:
- Single-file edits
- Independent utility creation
- Documentation updates
- Simple bug fixes

## Build Troubleshooting

If build fails:
1. Analyze error messages
2. Fix incrementally
3. Verify after each fix

## Data Pipeline Performance

For this project's data pipeline:
- Use Polars lazy evaluation for large datasets
- Use threading for concurrent HTTP requests (already in BaseProduct)
- Avoid loading entire JSONL files into memory when streaming is possible
- Use `pl.scan_ndjson()` for lazy reading of large files
