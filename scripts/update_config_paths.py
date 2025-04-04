#!/usr/bin/env python
"""
A script to update file paths in configuration files.
This ensures all the paths are consistent and point to the correct data directories.
"""

from pathlib import Path

import yaml


def update_config_path(config_file: str, field_path: str, old_prefix: str, new_prefix: str):
    """Update a path in a configuration file.

    Args:
        config_file: Path to the YAML config file
        field_path: Path to the field in the YAML (dot-separated)
        old_prefix: Old path prefix to replace
        new_prefix: New path prefix to use
    """
    # Read the YAML file
    with open(config_file, "r") as f:
        config = yaml.safe_load(f) or {}

    # Navigate to the field
    parts = field_path.split(".")
    current = config
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]

    last_part = parts[-1]
    if last_part in current and isinstance(current[last_part], str):
        path = current[last_part]
        if path.startswith(old_prefix):
            current[last_part] = path.replace(old_prefix, new_prefix, 1)
            print(f"Updated {field_path} from '{path}' to '{current[last_part]}'")
        else:
            print(f"Field {field_path} doesn't start with '{old_prefix}', skipping")
    else:
        print(f"Field {field_path} not found or not a string, skipping")

    # Write the updated YAML file
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def update_all_configs():
    """Update all development configuration files."""
    # Get the base directory
    base_dir = Path(__file__).parent.parent

    # Path to config files directory
    config_dir = base_dir / "src" / "clustering" / "dagster" / "resources" / "configs"

    # Find all YAML files
    yaml_files = list(config_dir.glob("*.yml"))

    if not yaml_files:
        print(f"No YAML files found in {config_dir}")
        return

    # Fields to update
    fields_to_update = [
        "readers.internal_sales.path",
        "readers.internal_need_state.path",
        "readers.external_sales.path",
        "writers.internal_sales_output.path",
        "writers.internal_sales_percent_output.path",
        "writers.internal_clusters_output.path",
        "writers.external_data_output.path",
        "writers.merged_clusters_output.path",
    ]

    # Update each YAML file
    for yaml_file in yaml_files:
        print(f"\nUpdating {yaml_file}...")
        for field in fields_to_update:
            update_config_path(str(yaml_file), field, "data/internal/", "data/raw/")


if __name__ == "__main__":
    update_all_configs()
    print("\nConfiguration update complete!")
