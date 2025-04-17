# Testing the Clustering Package

This directory contains tests for the clustering package.

## Structure

- `core/`: Unit tests for core components (schemas, models, utils)
- `io/`: Tests for IO functionality (readers, writers, blob storage, etc.)
- `dagster/`: Tests for Dagster-specific components (assets, jobs, resources)
- `integration/`: Integration tests for workflow combinations

## Running Tests

To run all tests:

```bash
make test
```

To run tests for a specific component:

```bash
pytest tests/core/test_schemas.py -v
```

To run tests with code coverage:

```bash
pytest tests/ --cov=src/clustering --cov-report=term --cov-report=html
```

## Writing Tests

When writing new tests:

1. Place tests in the appropriate directory based on component type
2. Use fixtures from `conftest.py` when possible
3. Follow existing naming conventions

## Test Data

Test data is generated using fixtures in `conftest.py`. If you need additional test data, consider adding a new fixture there.

## Mocking

Mock external dependencies (like Azure Blob Storage) using pytest fixtures. See the `mock_env_vars` fixture in `conftest.py` for an example.
