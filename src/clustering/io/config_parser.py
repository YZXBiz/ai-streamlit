"""Configuration management for clustering applications.

This module provides a unified approach to configuration management
with support for layered configurations, environment-specific overrides,
and validation against schemas.
"""

import os
from pathlib import Path
from typing import Any, ClassVar, dict

import yaml
from loguru import logger


class ConfigManager:
    """Configuration manager using the Singleton pattern.

    Provides centralized configuration loading, validation, and access.
    Supports layered configuration with environment-specific overrides.
    """

    _instance: ClassVar[dict[str, "ConfigManager"]] = {}

    @classmethod
    def get_instance(cls, config_dir: str | Path | None = None) -> "ConfigManager":
        """Get or create a ConfigManager instance.

        Args:
            config_dir: Path to the configuration directory
                (defaults to 'configs' relative to the project root)

        Returns:
            A ConfigManager instance
        """
        config_dir = str(config_dir or "configs")
        if config_dir not in cls._instance:
            cls._instance[config_dir] = cls(config_dir)
        return cls._instance[config_dir]

    def __init__(self, config_dir: str | Path):
        """Initialize the ConfigManager.

        Args:
            config_dir: Path to the configuration directory
        """
        self.config_dir = Path(config_dir)
        self._config: dict[str, Any] = {}
        self._load_default_config()

    def _load_default_config(self) -> None:
        """Load the default configuration from defaults.yml."""
        default_config_path = self.config_dir / "defaults.yml"

        # Fall back to base.yml if defaults.yml doesn't exist (for backward compatibility)
        if not default_config_path.exists():
            default_config_path = self.config_dir / "base.yml"

        if default_config_path.exists():
            with open(default_config_path, "r") as f:
                self._config = yaml.safe_load(f) or {}
            logger.debug(f"Loaded default configuration from {default_config_path}")
        else:
            logger.warning(f"Default configuration file not found at {default_config_path}")

    def load_environment_config(self, env: str = None) -> None:
        """Load environment-specific configuration.

        Args:
            env: Environment name (dev, prod, etc.)
                If None, uses the value from ENVIRONMENT env var, defaulting to 'dev'
        """
        if env is None:
            env = os.environ.get("ENVIRONMENT", "dev")

        env_config_path = self.config_dir / "environments" / f"{env}.yml"

        # For backward compatibility, check old paths too
        legacy_paths = [
            self.config_dir / env / "config.yml",
            self.config_dir / f"{env}.yml",
            self.config_dir / "env" / f"{env}.yml",
        ]

        # Try the new path first, then fall back to legacy paths
        found = False
        if env_config_path.exists():
            with open(env_config_path, "r") as f:
                env_config = yaml.safe_load(f) or {}
                self._update_config(env_config)
            logger.debug(f"Loaded {env} environment configuration from {env_config_path}")
            found = True
        else:
            for path in legacy_paths:
                if path.exists():
                    with open(path, "r") as f:
                        env_config = yaml.safe_load(f) or {}
                        self._update_config(env_config)
                    logger.debug(f"Loaded {env} environment configuration from {path}")
                    found = True
                    break

        if not found:
            logger.warning(f"Environment configuration for '{env}' not found")

    def load_component_config(self, component: str) -> None:
        """Load component-specific configuration.

        Args:
            component: Component name (clustering, preprocessing, io, etc.)
        """
        component_config_path = self.config_dir / "components" / f"{component}.yml"

        # For backward compatibility, check old paths too
        legacy_paths = [
            self.config_dir / f"{component}.yml",
            self.config_dir / f"internal_{component}.yml",
            self.config_dir / f"external_{component}.yml",
        ]

        # Try the new path first, then fall back to legacy paths
        found = False
        if component_config_path.exists():
            with open(component_config_path, "r") as f:
                component_config = yaml.safe_load(f) or {}
                self._update_config({component: component_config})
            logger.debug(f"Loaded {component} component configuration from {component_config_path}")
            found = True
        else:
            for path in legacy_paths:
                if path.exists():
                    with open(path, "r") as f:
                        component_config = yaml.safe_load(f) or {}
                        if component not in component_config and path.stem.startswith(("internal_", "external_")):
                            # For legacy files like internal_clustering.yml, assume the whole file is the component config
                            self._update_config({component: component_config})
                        else:
                            self._update_config(component_config)
                    logger.debug(f"Loaded {component} component configuration from {path}")
                    found = True
                    break

        if not found:
            logger.warning(f"Component configuration for '{component}' not found")

    def _update_config(self, new_config: dict) -> None:
        """Recursively update the configuration.

        Args:
            new_config: New configuration to merge
        """
        self._config = self._deep_update(self._config, new_config)

    def _deep_update(self, d: dict, u: dict) -> dict:
        """Recursively update a dictionary.

        Args:
            d: Dictionary to update
            u: Dictionary with updates

        Returns:
            Updated dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = self._deep_update(d[k], v)
            else:
                d[k] = v
        return d

    def get(self, key: str = None, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: Configuration key (dot notation supported, e.g., 'clustering.algorithm')
                If None, returns the entire configuration
            default: Default value if key is not found

        Returns:
            Configuration value or default
        """
        if key is None:
            return self._config

        parts = key.split(".")
        value = self._config

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default

        return value
