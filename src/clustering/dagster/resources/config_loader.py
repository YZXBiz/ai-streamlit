"""Configuration management resource for Dagster pipelines."""

import os
from typing import Any, Dict

import dagster as dg
import yaml
from pydantic import BaseModel, Field

from clustering.core.schemas import ClusteringConfig


class ConfigLoaderSchema(BaseModel):
    """Schema for configuration loader parameters."""

    env: str = Field("dev", description="Environment (dev, staging, prod)")
    config_dir: str = Field("configs", description="Directory containing configuration files")


@dg.resource(config_schema=ConfigLoaderSchema.model_json_schema())
def config_loader(context: dg.InitResourceContext) -> Dict[str, Any]:
    """Resource that loads configuration from YAML with environment awareness.

    Loads configurations in the following order, with later configs overriding earlier ones:
    1. Base config from configs/base/[config_name].yml
    2. Environment config from configs/env/[env]/[config_name].yml
    3. Specific config from configs/[config_name].yml (if it exists)

    Args:
        context: The Dagster resource initialization context

    Returns:
        Dict containing merged configuration
    """
    env = context.resource_config["env"]
    config_dir = context.resource_config["config_dir"]

    def load_yaml(path: str) -> Dict[str, Any]:
        """Load YAML file if it exists."""
        if not os.path.exists(path):
            context.log.debug(f"Config file not found: {path}")
            return {}

        with open(path, "r") as f:
            context.log.debug(f"Loading config from {path}")
            return yaml.safe_load(f) or {}

    def load_config(config_name: str) -> Dict[str, Any]:
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

    # Create a config object with methods to load different configs
    class ConfigAccessor:
        def load(self, config_name: str) -> Dict[str, Any]:
            """Load configuration by name."""
            return load_config(config_name)

        def get_env(self) -> str:
            """Get current environment."""
            return env

    return ConfigAccessor()
