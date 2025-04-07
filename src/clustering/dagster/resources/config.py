"""Resources for managing configuration in Dagster pipelines."""

import os
from types import SimpleNamespace

import dagster as dg
import yaml


@dg.resource(
    config_schema={
        "env": dg.Field(dg.String, default_value="dev", description="Environment name"),
    }
)
def simple_config(context: dg.InitResourceContext) -> SimpleNamespace:
    """Resource that loads configuration from a single YAML file.

    Args:
        context: The context for initializing the resource.

    Returns:
        SimpleNamespace: An object with attributes from the config file.
    """
    env = context.resource_config["env"]

    # Load environment config from a single file
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    env_config_path = os.path.join(config_dir, f"{env}.yml")

    if not os.path.exists(env_config_path):
        raise FileNotFoundError(f"Configuration file not found: {env_config_path}")

    with open(env_config_path) as f:
        config_data = yaml.safe_load(f)

    # Extract job parameters from config
    job_params = config_data.get("job_params", {})

    # Return as SimpleNamespace for attribute access
    return SimpleNamespace(**job_params)
