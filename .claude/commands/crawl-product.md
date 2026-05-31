---
name: crawl-product
description: Add or modify a crawler product in the vietlott-data pipeline.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /crawl-product

Workflow for adding or modifying a crawler product in the vietlott-data pipeline.

## Goal

Add a new product or modify an existing product's crawler, following the config-first approach.

## Key Files

- `src/vietlott/config/products.py` -- ProductConfig definitions
- `src/vietlott/config/map_class.py` -- Product name to crawler class mapping
- `src/vietlott/crawler/products/base.py` -- BaseProduct base class
- `src/vietlott/crawler/products/<product>.py` -- Product-specific crawler
- `src/vietlott/crawler/schema/requests.py` -- Request body schemas

## Suggested Sequence

1. Add ProductConfig in `products.py`
2. Create crawler class extending BaseProduct
3. Add request body schema in `requests.py`
4. Register in `map_class.py`
5. Add tests in `src/vietlott/tests/`
6. Run: `uv run pytest src/vietlott/tests`
7. Test crawl: `vietlott-crawl <product_name>`

## Conventions

- Override `process_result()` to parse HTML response
- Use `attrs` for request body schemas
- Use `BeautifulSoup` + `lxml` for HTML parsing
- Return list of dicts from `process_result()`
