"""Configuration management resource for Dagster pipelines."""

import os
from collections.abc import Callable
from typing import Any

import dagster as dg
import yaml  # type: ignore # We know yaml exists at runtime
from pydantic import BaseModel, Field

from clustering.core.schemas import ClusteringConfig


class ConfigLoaderSchema(BaseModel):
    """Schema for configuration loader parameters."""

    env: str = Field("dev", description="Environment (dev, staging, prod)")
    config_dir: str = Field("configs", description="Directory containing configuration files")


class ConfigAccessor:
    """Class for accessing configuration."""

    def __init__(self, load_fn: Callable[[str], dict[str, Any]], env: str):
        """Initialize the config accessor.

        Args:
            load_fn: Function to load configuration
            env: Environment name
        """
        self._load_fn = load_fn
        self._env = env

    def load(self, config_name: str) -> dict[str, Any]:
        """Load configuration by name.

        Args:
            config_name: Name of the configuration to load

        Returns:
            Dictionary of configuration values
        """
        return self._load_fn(config_name)

    def get_env(self) -> str:
        """Get current environment.

        Returns:
            Environment name
        """
        return self._env


@dg.resource(config_schema=ConfigLoaderSchema.model_json_schema())
def config_loader(context: dg.InitResourceContext) -> ConfigAccessor:
    """Resource that loads configuration from YAML with environment awareness.

    Loads configurations in the following order, with later configs overriding earlier ones:
    1. Base config from configs/base/[config_name].yml
    2. Environment config from configs/env/[env]/[config_name].yml
    3. Specific config from configs/[config_name].yml (if it exists)

    Args:
        context: The Dagster resource initialization context

    Returns:
        ConfigAccessor: Object for accessing configuration
    """
    env = context.resource_config["env"]
    config_dir = context.resource_config["config_dir"]
    logger = context.log

    def load_yaml(path: str) -> dict[str, Any]:
        """Load YAML file if it exists."""
        if not os.path.exists(path):
            # Safe access to logger which might be None
            if logger is not None:
                logger.debug(f"Config file not found: {path}")
            return {}

        with open(path) as f:
            # Safe access to logger which might be None
            if logger is not None:
                logger.debug(f"Loading config from {path}")
            return yaml.safe_load(f) or {}

    def load_config(config_name: str) -> dict[str, Any]:
        """Load and merge configuration files for a specific config."""
        # Base config
        base_path = os.path.join(config_dir, "base", f"{config_name}.yml")
        base_config = load_yaml(base_path)

        # Environment-specific config
        env_path = os.path.join(config_dir, "env", env, f"{config_name}.yml")
        env_config = load_yaml(env_path)

        # Specific config (optional override)
        specific_path = os.path.join(config_dir, f"{config_name}.yml")
        specific_config = load_yaml(specific_path)

        # Merge configs with priority: base < env < specific
        merged = {**base_config, **env_config, **specific_config}

        # Validate specific sections using Pydantic models
        if "clustering" in merged:
            merged["clustering"] = ClusteringConfig(**merged["clustering"]).model_dump()

        return merged

    # Create and return the configuration accessor
    return ConfigAccessor(load_config, env)
