#!/usr/bin/env python
"""
Script to migrate configuration files from the old structure to the new Dagster structure.

This script copies relevant configuration files from the root-level /configs directory
to the Dagster-specific configuration structure in src/clustering/dagster/resources/configs.
"""

import os
import shutil
import sys

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# File mappings - old path -> new path
FILE_MAPPINGS = {
    "configs/internal_clustering.yml": "src/clustering/dagster/resources/configs/job_configs/internal_clustering.yml",
    "configs/external_clustering.yml": "src/clustering/dagster/resources/configs/job_configs/external_clustering.yml",
    "configs/internal_preprocessing.yml": "src/clustering/dagster/resources/configs/job_configs/internal_preprocessing.yml",
    "configs/external_preprocessing.yml": "src/clustering/dagster/resources/configs/job_configs/external_preprocessing.yml",
    "configs/merge_int_ext.yml": "src/clustering/dagster/resources/configs/job_configs/merge_int_ext.yml",
}

# Directory to create
DIRECTORIES = [
    "src/clustering/dagster/resources/configs/job_configs",
]


def create_directories():
    """Create required directories if they don't exist."""
    for directory in DIRECTORIES:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def copy_files():
    """Copy files from old structure to new structure."""
    for old_path, new_path in FILE_MAPPINGS.items():
        if os.path.exists(old_path):
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(new_path), exist_ok=True)

            # Copy the file
            shutil.copy2(old_path, new_path)
            print(f"Copied: {old_path} -> {new_path}")
        else:
            print(f"Warning: Source file not found: {old_path}")


def main():
    """Main function to run the migration."""
    print("Starting configuration migration...")

    # Create directories
    create_directories()

    # Copy files
    copy_files()

    print("\nMigration completed. Please update your code to use the new configuration structure.")
    print("The old configuration files in /configs have been copied but not removed.")
    print("Once you've verified everything works, you can safely remove the old files.")


if __name__ == "__main__":
    main()
