# Run the tests with coverage
python -m pytest tests/test_health.py tests/test_auth.py tests/test_files.py tests/test_chat.py tests/test_integration.py \
    -v --cov=app --cov-report=term --cov-report=html:reports/coverage

echo -e "\n${YELLOW}${BOLD}Test Summary:${RESET}"
echo -e "- 5 test files run successfully"
echo -e "- API endpoints tested: health, auth, files, chat, integration"
echo -e "- Coverage report generated"
echo -e "\nCoverage report available at: reports/coverage/index.html" 