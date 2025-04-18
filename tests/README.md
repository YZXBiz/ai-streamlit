# Testing Organization for Multi-Package Structure

This directory contains tests for the clustering project's multi-package architecture.

## Test Structure

### Package-Based Organization

Tests are organized to mirror our multi-package structure:

- `clustering-shared/`: Tests for shared functionality used across packages
  - `core/`: Core domain models, utilities, and common functionality
  - `io/`: IO operations, readers, writers, data access

- `clustering-pipeline/`: Tests for pipeline-specific functionality
  - `dagster/`: Dagster assets, jobs, resources, and pipeline definitions
  - `models/`: Machine learning models, algorithms, and related components

- `clustering-cli/`: Tests for CLI tools and command-line interfaces

### Test Types

- `unit/`: Focused tests for individual components (default location for most tests)
- `integration/`: Tests that verify interactions between multiple components
- `e2e/`: End-to-end tests that verify complete workflows

## Running Tests

You can run tests for specific packages or components:

```bash
# Run all tests
make test

# Run tests for a specific package
make test-shared
make test-pipeline
make test-cli

# Run specific test types
make test-unit
make test-integration
```

## Writing Tests

1. Place tests in the appropriate package directory
2. Use fixtures from the package-specific `conftest.py` when possible
3. Follow package-specific patterns and conventions
4. Name test files with `test_` prefix

## Fixtures and Common Utilities

- Root-level `conftest.py` contains project-wide fixtures
- Each package directory has its own `conftest.py` for package-specific fixtures

## Test Data

Test data is generated using fixtures in `conftest.py`. If you need additional test data, consider adding a new fixture there.

## Mocking

Mock external dependencies (like Azure Blob Storage) using pytest fixtures. See the `mock_env_vars` fixture in `conftest.py` for an example.
