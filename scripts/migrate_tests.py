#!/usr/bin/env python
"""Script to migrate tests from old directory structure to new package-based structure.

This script helps reorganize tests from the old flat structure into the new 
package-based structure. It analyzes test files and moves them to the appropriate 
location in the new structure.

Usage:
    uv run scripts/migrate_tests.py

Note: This script should be run from the project root directory.
"""

import os
import re
import shutil
from pathlib import Path
from typing import Optional


# Mapping of old directories to new package-based directories
DIRECTORY_MAPPING = {
    "tests/core": {
        "default": "tests/clustering-shared/unit/core",
        "patterns": {
            r"test_internal_": "tests/clustering-pipeline/unit",
            r"test_external_": "tests/clustering-pipeline/unit",
        }
    },
    "tests/io": {
        "default": "tests/clustering-shared/unit/io",
        "patterns": {}
    },
    "tests/dagster": {
        "default": "tests/clustering-pipeline/dagster",
        "patterns": {}
    },
    "tests/integration": {
        "default": "tests/integration",
        "patterns": {
            r"test_job_workflow": "tests/clustering-pipeline/integration",
        }
    },
}


def determine_target_directory(file_path: Path) -> str:
    """Determine the target directory for a test file.
    
    Args:
        file_path: Path to the test file.
        
    Returns:
        str: Path to the target directory.
    """
    old_dir = str(file_path.parent)
    filename = file_path.name
    
    if old_dir in DIRECTORY_MAPPING:
        mapping = DIRECTORY_MAPPING[old_dir]
        
        # Check if file matches any specific pattern
        for pattern, target_dir in mapping["patterns"].items():
            if re.search(pattern, filename):
                return target_dir
        
        # Use default if no pattern matches
        return mapping["default"]
    
    # If no mapping exists, keep in the same directory
    return old_dir


def migrate_test_file(file_path: Path, dry_run: bool = True) -> Optional[Path]:
    """Migrate a test file to the new directory structure.
    
    Args:
        file_path: Path to the test file.
        dry_run: If True, only print what would be done.
        
    Returns:
        Optional[Path]: Path to the new file location if migrated, None otherwise.
    """
    if not file_path.is_file() or not file_path.name.startswith("test_"):
        return None
    
    target_dir = determine_target_directory(file_path)
    target_path = Path(target_dir) / file_path.name
    
    # Create target directory if it doesn't exist
    if not dry_run:
        os.makedirs(Path(target_dir), exist_ok=True)
    
    print(f"Moving {file_path} -> {target_path}")
    
    if not dry_run:
        shutil.copy2(file_path, target_path)
        return target_path
    
    return None


def migrate_tests(dry_run: bool = True) -> None:
    """Migrate all test files to the new structure.
    
    Args:
        dry_run: If True, only print what would be done.
    """
    test_dirs = ["tests/core", "tests/io", "tests/dagster", "tests/integration"]
    
    for test_dir in test_dirs:
        if not Path(test_dir).exists():
            print(f"Warning: Directory {test_dir} does not exist, skipping.")
            continue
        
        for file_path in Path(test_dir).glob("test_*.py"):
            migrate_test_file(file_path, dry_run)
    
    print("\nMigration", "simulation" if dry_run else "completed", "successfully!")
    if dry_run:
        print("This was a dry run. No files were actually moved.")
        print("Run with --execute to perform the actual migration.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate tests to new package-based structure")
    parser.add_argument("--execute", action="store_true", help="Execute the migration (default is dry run)")
    args = parser.parse_args()
    
    migrate_tests(dry_run=not args.execute) 