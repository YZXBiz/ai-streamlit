#!/bin/bash

# Run backend tests with coverage reporting
set -e

echo "==> Setting up test environment..."
export PYTHONPATH=.
export ENV=test
mkdir -p logs

echo "==> Running backend tests with coverage..."
uv run -m pytest backend/tests -v --cov=backend --cov-report=term --cov-report=html:reports/coverage

# Check if tests passed
if [ $? -eq 0 ]; then
    echo "✅ All tests passed!"
    echo "Coverage report generated in reports/coverage/"
else
    echo "❌ Tests failed!"
    exit 1
fi

# Run specific integration tests if requested
if [ "$1" == "integration" ]; then
    echo "==> Running integration tests only..."
    uv run -m pytest backend/tests/test_integration.py -v
fi 