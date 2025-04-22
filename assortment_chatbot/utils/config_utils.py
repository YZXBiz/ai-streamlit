"""
Configuration utilities for the assortment_chatbot application.

This module provides functions to load and manage configuration
settings for the assortment_chatbot application.
"""

import json
import os
from pathlib import Path
from typing import Any

import toml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_env_var(key: str, default: str | None = None) -> str:
    """
    Get an environment variable value.

    Args:
        key: The environment variable name
        default: Default value if environment variable is not set

    Returns:
        The environment variable value or default

    Raises:
        ValueError: If the environment variable is not set and no default is provided
    """
    value = os.environ.get(key, default)
    if value is None:
        raise ValueError(f"Environment variable {key} is not set and no default provided")
    return value


def load_config_file(file_path: str | Path, config_format: str = "auto") -> dict[str, Any]:
    """
    Load a configuration file.

    Args:
        file_path: Path to the configuration file
        config_format: Format of the configuration file ('json', 'toml', or 'auto' to detect from extension)

    Returns:
        Dictionary containing the configuration

    Raises:
        FileNotFoundError: If the configuration file does not exist
        ValueError: If the file format is not supported
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    # Determine format from extension if auto
    if config_format == "auto":
        extension = file_path.suffix.lower()
        if extension == ".json":
            config_format = "json"
        elif extension == ".toml":
            config_format = "toml"
        else:
            raise ValueError(f"Unsupported configuration file extension: {extension}")

    # Load based on format
    if config_format == "json":
        with open(file_path) as f:
            return json.load(f)
    elif config_format == "toml":
        return toml.load(file_path)
    else:
        raise ValueError(f"Unsupported configuration format: {config_format}")


def get_app_config() -> dict[str, Any]:
    """
    Get the application configuration from environment variables
    and configuration files.

    Returns:
        Dictionary containing the application configuration
    """
    # Base configuration from environment variables
    config = {
        "app": {
            "debug": os.environ.get("DEBUG", "false").lower() == "true",
            "environment": os.environ.get("ENVIRONMENT", "development"),
            "log_level": os.environ.get("LOG_LEVEL", "INFO"),
        },
        "data": {
            "data_dir": os.environ.get("DATA_DIR", "data"),
            "max_upload_size_mb": int(os.environ.get("MAX_UPLOAD_SIZE_MB", "100")),
            "allowed_extensions": os.environ.get("ALLOWED_EXTENSIONS", ".csv,.xlsx,.json").split(
                ","
            ),
        },
        "database": {
            "use_snowflake": os.environ.get("USE_SNOWFLAKE", "false").lower() == "true",
            "snowflake_account": os.environ.get("SNOWFLAKE_ACCOUNT", ""),
            "snowflake_user": os.environ.get("SNOWFLAKE_USER", ""),
            "snowflake_warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE", ""),
            "snowflake_database": os.environ.get("SNOWFLAKE_DATABASE", ""),
            "use_duckdb": os.environ.get("USE_DUCKDB", "true").lower() == "true",
            "duckdb_path": os.environ.get("DUCKDB_PATH", ":memory:"),
        },
        "ai": {
            "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
            "use_ai_features": os.environ.get("USE_AI_FEATURES", "false").lower() == "true",
            "ai_model": os.environ.get("AI_MODEL", "gpt-3.5-turbo"),
        },
    }

    # Check for and load optional config file
    config_file = os.environ.get("CONFIG_FILE")
    if config_file:
        try:
            file_config = load_config_file(config_file)
            # Merge file config with environment config (env vars take precedence)
            _merge_configs(config, file_config)
        except Exception:
            pass

    return config


def _merge_configs(base_config: dict, override_config: dict) -> None:
    """
    Merge two configuration dictionaries recursively.

    Args:
        base_config: Base configuration that will be updated
        override_config: Configuration with values to override
    """
    for key, value in override_config.items():
        if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
            _merge_configs(base_config[key], value)
        else:
            base_config[key] = value


def save_config(config: dict[str, Any], file_path: str | Path, format: str = "json") -> None:
    """
    Save configuration to a file.

    Args:
        config: Configuration dictionary to save
        file_path: Path to save the configuration to
        format: Format to save the configuration in ('json' or 'toml')

    Raises:
        ValueError: If the format is not supported
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(file_path, "w") as f:
            json.dump(config, f, indent=2)
    elif format == "toml":
        with open(file_path, "w") as f:
            toml.dump(config, f)
    else:
        raise ValueError(f"Unsupported configuration format: {format}")
