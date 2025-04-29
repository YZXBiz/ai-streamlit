# Backend Tests

This directory contains tests for the PandasAI Chat Application backend API.

## Test Structure

The tests are organized by feature/endpoint:

- `test_health.py` - Tests for the health check endpoint
- `test_auth.py` - Tests for authentication endpoints (login, register)
- `test_files.py` - Tests for file management endpoints (upload, retrieval)
- `test_chat.py` - Tests for chat functionality (sessions, messages, queries)
- `test_integration.py` - End-to-end flow tests (currently not used due to mocking issues)

## Running Tests

### Using Makefile

The easiest way to run tests is using the provided Makefile targets:

```bash
# Run the individual endpoint tests with coverage
make test-backend-integration

# Run all backend tests (including integration tests)
make test-backend
```

### Using Custom Scripts

You can also run tests using the provided scripts:

```bash
# Run specific test files with coverage reporting
cd backend
./run_tests_with_coverage.sh

# Run tests using Python script
cd backend
python tests/run_tests.py
```

### Running Individual Tests

To run a specific test file directly:

```bash
pytest backend/tests/test_health.py -v
```

## Test Configuration

Tests use a mock configuration defined in `conftest.py`. This includes:

- Mock settings for database, API keys, etc.
- Mock service implementations
- Mock dependencies

This approach allows tests to run without actual database connections or external API calls.

## Coverage Reporting

Test coverage reports are generated in HTML format at `backend/reports/coverage/index.html` when running tests with coverage enabled.

## Adding New Tests

When adding new tests:

1. Consider which existing test file your test belongs in, or create a new file for distinct functionality
2. Use the provided fixtures and mock services in your tests
3. Add your new test file to the `test_files` list in `run_tests.py` if it's a separate file
4. Make sure to run with coverage to ensure your tests are adequately covering the code

## Test Fixtures

Common test fixtures are defined in `conftest.py` and individual test files:

- `client` - FastAPI TestClient instance
- `auth_headers` - Authentication headers for protected endpoints

## Mock Dependencies

Tests use mock implementations of services to avoid actual database calls or external API requests. This includes:

- `MockAuthService` - For authentication operations
- `MockFileService` - For file operations
- `MockChatService` - For chat operations

## Environment Variables

Tests set environment variables for testing in the test scripts:

```
TESTING=true
DB_PORT=5432
OPENAI_API_KEY=sk-dummy-key-for-testing
```

## Current Test Status (April 2024)

The tests are currently at approximately 43% coverage. Here's the breakdown:

- **Domain Models**: ~86% coverage (almost complete)
- **Port Interfaces**: ~75% coverage (good progress)
- **API Integration Tests**: Partial coverage (~65% for some modules)
- **Adapters**: Low coverage (~20-30% for most adapters)
- **Services**: Low coverage (~20-30% for most services)

## Path to 100% Coverage

To achieve 100% coverage, the following needs to be addressed:

1. **API Integration Tests**:
   - Fix authentication token handling in tests
   - Resolve URL path discrepancies (e.g., `/api/files/` vs `/api/v1/files/`)
   - Update Message model tests to match the current implementation (remove 'role' field)

2. **Adapter Tests**:
   - Add comprehensive tests for each adapter implementation (db_postgres, db_duckdb, storage_local, etc.)
   - Create mock external services for testing adapters

3. **Service Tests**:
   - Add unit tests for all service methods
   - Create mock repositories/adapters for testing services

4. **Fix Imports**:
   - Resolve import issues (e.g., `backend.dataframe.collection` in test_collection.py)

5. **Test Execution**:
   - Update test runner scripts to execute all tests correctly

When adding new tests, remember to:
- Use the existing patterns and fixtures
- Ensure each test is focused on a single functionality
- Mock external dependencies
- Include both success and failure cases 