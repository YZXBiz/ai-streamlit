#!/usr/bin/env python3
"""
Script to fix import paths in test files, changing 'from app.' to 'from backend.app.'
"""

import os
import re
from pathlib import Path


def fix_imports_in_file(file_path):
    """Replace 'from app.' and 'import app' with 'from backend.app.' and 'import backend.app'"""
    print(f"Processing {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Fix imports
    modified_content = re.sub(r"from app\.", r"from backend.app.", content)
    modified_content = re.sub(r"import app\.", r"import backend.app.", modified_content)

    # Fix duplicated assertions (common syntax error)
    modified_content = re.sub(r"(assert .+?)\s+\1", r"\1", modified_content)

    if content != modified_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(modified_content)
        print(f"Fixed imports in {file_path}")
    else:
        print(f"No changes needed in {file_path}")


def fix_imports_in_directory(directory):
    """Fix imports in all .py files in the directory and its subdirectories"""
    test_files = Path(directory).glob("**/*.py")
    for file_path in test_files:
        if ".venv" not in str(file_path):  # Skip virtual environment
            fix_imports_in_file(file_path)


if __name__ == "__main__":
    # Fix imports in all test files
    fix_imports_in_directory("backend/tests")
    print("Import paths fixed. Tests should now be able to find the modules.")
