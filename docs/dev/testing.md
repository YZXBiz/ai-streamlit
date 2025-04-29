# Testing Strategy

This document outlines the testing strategy for the PandasAI Chat Application.

## Overview

We use a multi-layered testing approach to ensure code quality and functionality:

1. **Unit Tests** - Testing individual components in isolation
2. **Integration Tests** - Testing interactions between components
3. **API Tests** - Testing HTTP endpoints
4. **End-to-End Tests** - Testing complete user flows

## Testing Technologies

- **Test Framework**: pytest
- **Coverage Reporting**: pytest-cov
- **API Testing**: FastAPI TestClient
- **Mocking**: unittest.mock, AsyncMock
- **Continuous Integration**: GitHub Actions

## Test Organization

Tests are organized by component and divided into individual test files:

```
backend/
├── tests/
│   ├── adapters/      # Tests for adapter implementations
│   ├── api/           # Tests for API endpoints
│   ├── domain/        # Tests for domain models
│   ├── ports/         # Tests for interface definitions
│   ├── services/      # Tests for service layer
│   ├── utils/         # Test utilities
│   ├── test_health.py # Health endpoint tests
│   ├── test_auth.py   # Authentication tests
│   ├── test_files.py  # File handling tests
│   ├── test_chat.py   # Chat functionality tests
│   └── conftest.py    # Test fixtures and configuration
```

## Test Configuration

Tests are configured to run with minimal external dependencies using mocks. The `conftest.py` file sets up:

- Mock database connections
- Mock services
- Mock external API calls

This allows tests to run quickly and without requiring external resources.

## Running Tests

### Using Make

```bash
# Run all backend tests
make test-backend

# Run integration tests only
make test-backend-integration

# Run all tests with validation
make validate-tests

# Run with coverage
make test-backend-coverage
```

### Using pytest directly

```bash
# Run all tests
pytest backend/tests

# Run specific test file
pytest backend/tests/test_auth.py

# Run with coverage
pytest backend/tests --cov=backend
```

## Coverage Reporting

Coverage reports are generated in HTML format at `backend/reports/coverage/index.html`. 

To generate a coverage report:

```bash
make test-backend-integration
```

## Test Mocking Approach

Our backend tests use mock implementations to isolate the tests from external dependencies:

1. **Service Mocks** - Mock implementations of services allow testing API endpoints without running actual business logic.

2. **Repository Mocks** - Mock implementations of repositories avoid database connections during testing.

3. **External API Mocks** - Mocks for external APIs (like OpenAI) prevent tests from making actual API calls.

## Continuous Integration

Tests are executed automatically via GitHub Actions on:

- Pull requests to main branch
- Direct pushes to main branch

See `.github/workflows/test.yml` for the CI configuration.

## Best Practices

When adding new code, follow these testing best practices:

1. **Write Tests First** - Consider using Test-Driven Development (TDD) for new features.

2. **Test Coverage** - Aim for high test coverage, especially for critical components.

3. **Meaningful Assertions** - Tests should have meaningful assertions that verify actual functionality.

4. **Isolation** - Tests should be independent and not rely on side effects from other tests.

5. **Clean Up** - Tests should clean up after themselves, especially if they create files or data.

## Test Categories

### 1. Unit Tests

Unit tests verify that individual components work correctly in isolation:

- Domain model tests
- Service method tests
- Adapter implementation tests

### 2. Integration Tests

Integration tests verify the interactions between components:

- Service to repository interactions
- API to service integration
- Database operations

### 3. API Tests

API tests verify that the API endpoints function correctly:

- Authentication endpoints
- File management endpoints
- Chat session endpoints
- Query endpoints

These tests use FastAPI's TestClient to simulate HTTP requests without running a server.

### 4. End-to-End Tests

End-to-end tests verify complete user flows from frontend to backend. Currently, we manually test end-to-end functionality. 

Future work includes adding automated E2E tests using tools like Playwright or Cypress. 