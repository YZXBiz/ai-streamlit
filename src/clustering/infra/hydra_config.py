"""Hydra-style configuration utilities for YAML files with environment variables."""

import os
import re
from pathlib import Path
from typing import Any, Dict, Union
import yaml


class OmegaConfLoader:
    """
    A simplified version of OmegaConf loader that handles environment variable interpolation
    in YAML files using the ${env:VAR,default} syntax.
    """

    ENV_VAR_PATTERN = re.compile(r"\${env:([^,}]+)(?:,([^}]+))?}")
    NESTED_VAR_PATTERN = re.compile(r"\${([^:{}]+(?:\.[^:{}]+)*)}")

    @classmethod
    def load(cls, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a YAML configuration file with Hydra-style environment variable interpolation.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            Dict containing the configuration with resolved variables
        """
        with open(config_path, "r") as f:
            # Load the raw YAML
            config = yaml.safe_load(f)

        if not config:
            return {}

        # Resolve all variables
        return cls._resolve_config(config)

    @classmethod
    def _resolve_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively resolve environment variables and references in a configuration dict.

        Args:
            config: Configuration dictionary

        Returns:
            Configuration with all variables resolved
        """
        result = {}

        # First pass: collect all values
        for key, value in config.items():
            if isinstance(value, dict):
                result[key] = cls._resolve_config(value)
            elif isinstance(value, list):
                result[key] = [
                    cls._resolve_config(item)
                    if isinstance(item, dict)
                    else cls._resolve_value(item, result)
                    for item in value
                ]
            else:
                result[key] = cls._resolve_value(value, result)

        return result

    @classmethod
    def _resolve_value(cls, value: Any, config: Dict[str, Any]) -> Any:
        """
        Resolve a single value, handling environment variables and nested references.

        Args:
            value: The value to resolve
            config: The configuration dictionary for nested references

        Returns:
            Resolved value
        """
        if not isinstance(value, str):
            return value

        # Process environment variables
        value = cls._resolve_env_vars(value)

        # Process nested references (${paths.base_dir} etc)
        value = cls._resolve_nested_vars(value, config)

        return value

    @classmethod
    def _resolve_env_vars(cls, value: str) -> str:
        """
        Resolve environment variables in a string using ${env:VAR,default} syntax.

        Args:
            value: String containing environment variable references

        Returns:
            String with environment variables replaced
        """
        if not isinstance(value, str) or "${env:" not in value:
            return value

        def _replace_env_var(match):
            var_name = match.group(1)
            default = match.group(2) if match.group(2) is not None else ""

            # Get the environment variable or use default
            return os.environ.get(var_name, default)

        # Replace all occurrences
        return cls.ENV_VAR_PATTERN.sub(_replace_env_var, value)

    @classmethod
    def _resolve_nested_vars(cls, value: str, config: Dict[str, Any]) -> str:
        """
        Resolve nested variable references (${paths.base_dir}).

        Args:
            value: String containing nested references
            config: Configuration dictionary to look up references

        Returns:
            String with references replaced
        """
        if not isinstance(value, str) or "${" not in value:
            return value

        def _get_from_nested_dict(keys, d):
            """Get value from nested dictionary using dot notation."""
            if not keys or not d:
                return None

            if len(keys) == 1:
                return d.get(keys[0])

            if keys[0] in d and isinstance(d[keys[0]], dict):
                return _get_from_nested_dict(keys[1:], d[keys[0]])

            return None

        def _replace_nested_var(match):
            path = match.group(1).split(".")
            result = _get_from_nested_dict(path, config)
            return str(result) if result is not None else match.group(0)

        # Replace all occurrences
        return cls.NESTED_VAR_PATTERN.sub(_replace_nested_var, value)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a configuration file with Hydra-style environment variable interpolation.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dict containing the configuration with resolved variables
    """
    return OmegaConfLoader.load(config_path)
