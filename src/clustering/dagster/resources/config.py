"""Resources for managing configuration in Dagster pipelines."""

import os
from types import SimpleNamespace

import dagster as dg
import yaml


@dg.resource(
    config_schema={
        "config_path": dg.Field(
            dg.String, is_required=True, description="Path to the configuration file"
        ),
        "env": dg.Field(dg.String, default_value="dev", description="Environment name"),
    }
)
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
    configs_dir = os.path.join(os.path.dirname(__file__), "configs")
    base_config_path = os.path.join(configs_dir, "base.yml")

    # Load base config if it exists
    base_config = {}
    if os.path.exists(base_config_path):
        with open(base_config_path) as f:
            base_config = yaml.safe_load(f)

    # Load environment-specific config if it exists
    env_config = {}
    env_config_path = os.path.join(configs_dir, f"{env}.yml")
    if os.path.exists(env_config_path):
        with open(env_config_path) as f:
            env_config = yaml.safe_load(f)

    # Load specific config
    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    # Merge configs with priority: base < env < specific
    merged_config = {**base_config, **env_config, **config_data}

    # Convert to SimpleNamespace for attribute access
    return SimpleNamespace(**merged_config)
