#!/usr/bin/env python
"""Script to run the individual test files for backend components."""

import os
import subprocess
import sys
from pathlib import Path

# Set environment variables for testing
os.environ["TESTING"] = "true"
os.environ["DB_PORT"] = "5432"
os.environ["OPENAI_API_KEY"] = "sk-dummy-key-for-testing"
os.environ["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])

# Test files to run (the ones we've verified work)
test_files = [
    "api/test_health.py",
    "api/test_auth_routes.py",
    "api/test_file_routes.py",
    "api/test_chat_routes.py",
    "api/test_integration.py",
]

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def main():
    """Run the test files."""
    print(f"{BOLD}Running backend tests{RESET}\n")

    tests_dir = Path(__file__).parent
    passed = 0
    failed = 0

    for test_file in test_files:
        test_path = tests_dir / test_file
        if not test_path.exists():
            print(f"{RED}Test file not found: {test_file}{RESET}")
            continue

        print(f"Running tests in {test_file}...")
        result = subprocess.run(
            ["pytest", "-xvs", str(test_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(f"{GREEN}PASSED{RESET}: {test_file}")
            passed += 1
        else:
            print(f"{RED}FAILED{RESET}: {test_file}")
            print(f"Error: {result.stderr}")
            failed += 1

        print()

    # Print summary
    print(f"{BOLD}Summary:{RESET}")
    print(f"Total test files: {len(test_files)}")
    print(f"Passed: {GREEN}{passed}{RESET}")
    print(f"Failed: {RED}{failed}{RESET}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
