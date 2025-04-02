"""Resources for managing configuration in Dagster pipelines."""

import os
from types import SimpleNamespace

import dagster as dg
import yaml
from pydantic import BaseModel


class ConfigSchema(BaseModel):
    """Schema for configuration resource."""

    config_path: str
    env: str = "dev"


@dg.resource(config_schema=ConfigSchema.model_json_schema())
def clustering_config(context: dg.InitResourceContext) -> SimpleNamespace:
    """Resource that loads clustering configuration from YAML.

    Args:
        context: The context for initializing the resource.

    Returns:
        SimpleNamespace: An object with attributes from the config file.
    """
    config_path = context.resource_config["config_path"]
    env = context.resource_config["env"]

    # First load base config
    base_config_path = os.path.join(os.path.dirname(config_path), "base.yml")

    # Load base config if it exists
    base_config = {}
    if os.path.exists(base_config_path):
        with open(base_config_path, "r") as f:
            base_config = yaml.safe_load(f)

    # Load environment-specific config if it exists
    env_config = {}
    env_config_path = os.path.join(os.path.dirname(config_path), env, os.path.basename(config_path))
    if os.path.exists(env_config_path):
        with open(env_config_path, "r") as f:
            env_config = yaml.safe_load(f)

    # Load specific config
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    # Merge configs with priority: base < env < specific
    merged_config = {**base_config, **env_config, **config_data}

    # Convert to SimpleNamespace for attribute access
    return SimpleNamespace(**merged_config)
