# Testing the Clustering Package

This directory contains tests for the clustering package.

## Structure

- `unit/`: Unit tests for individual components
- `integration/`: Integration tests for workflow combinations

## Running Tests

To run all tests:

```bash
make test
```

To run a specific test file:

```bash
pytest tests/unit/test_configs.py -v
```

To run tests with code coverage:

```bash
pytest tests/ --cov=src/clustering --cov-report=term --cov-report=html
```

## Writing Tests

When writing new tests:

1. Place unit tests in the appropriate file under `tests/unit/`
2. Place integration tests in the appropriate file under `tests/integration/`
3. Use fixtures from `conftest.py` when possible
4. Follow the existing naming conventions

## Test Data

Test data is generated using fixtures in `conftest.py`. If you need additional test data, consider adding a new fixture there.

## Mocking

Mock external dependencies (like Azure Blob Storage) using pytest fixtures. See the `mock_env_vars` fixture in `conftest.py` for an example.
