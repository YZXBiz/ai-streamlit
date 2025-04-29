#!/bin/bash

# Run the backend tests with coverage reporting
# This script runs the individual test files with pytest coverage

# Set environment variables for testing
export TESTING=true
export DB_PORT=5432
export OPENAI_API_KEY="sk-dummy-key-for-testing"
export PYTHONPATH=$(pwd)/../

# Create a directory for reports
mkdir -p reports/coverage

# Colors for output
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
BOLD="\033[1m"
RESET="\033[0m"

echo -e "${BOLD}Running backend integration tests with coverage${RESET}\n"

# Run the tests with coverage
python -m pytest tests/api/test_health.py tests/api/test_auth_routes.py tests/api/test_file_routes.py tests/api/test_chat_routes.py tests/api/test_integration.py \
    -v --cov=app --cov-report=term --cov-report=html:reports/coverage

# Check if tests passed
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}${BOLD}All tests passed!${RESET}"
    echo -e "\nCoverage report saved to reports/coverage/index.html"
else
    echo -e "\n${RED}${BOLD}Tests failed!${RESET}"
    exit 1
fi

echo -e "\n${YELLOW}${BOLD}Test Summary:${RESET}"
echo -e "- 5 test files run successfully"
echo -e "- API endpoints tested: health, auth, files, chat, integration"
echo -e "- Coverage report generated"
echo -e "\nCoverage report available at: reports/coverage/index.html" 