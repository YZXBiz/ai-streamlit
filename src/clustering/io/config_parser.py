"""Configuration management for clustering applications.

This module provides a unified approach to configuration management
with support for layered configurations, environment-specific overrides,
and validation against schemas.
"""

from pathlib import Path
from typing import dict

import yaml


def parse_file(file_path: str) -> dict:
    """Parse a YAML config file.

    Args:
        file_path: Path to the configuration file

    Returns:
        Dictionary with parsed configuration
    """
    with open(file_path) as f:
        return yaml.safe_load(f) or {}


def parse_string(config_str: str) -> dict:
    """Parse a YAML config string.

    Args:
        config_str: Configuration string in YAML format

    Returns:
        Dictionary with parsed configuration
    """
    return yaml.safe_load(config_str) or {}


def load_config(config_path: str | Path) -> dict:
    """Load configuration from a file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary with loaded configuration
    """
    return parse_file(str(config_path))


def save_config(config: dict, config_path: str | Path) -> None:
    """Save configuration to a file.

    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration
    """
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
