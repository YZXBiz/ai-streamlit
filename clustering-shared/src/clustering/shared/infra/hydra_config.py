"""Parse, merge, and convert config objects."""

import os
import typing as T

import omegaconf as oc

# %% TYPES

Config = oc.ListConfig | oc.DictConfig

# %% PARSERS


def parse_file(path: str) -> Config | None:
    """Parse a config file from a path.

    Args:
        path: Path to local config.

    Returns:
        Config representation of the config file or None if file doesn't exist.
    """
    if not os.path.exists(path):
        return None
    return oc.OmegaConf.load(path)


def parse_string(string: str) -> Config:
    """Parse the given config string.

    Args:
        string: Content of config string.

    Returns:
        Config representation of the config string.
    """
    return oc.OmegaConf.create(string)


# %% MERGERS


def merge_configs(configs: T.Sequence[Config]) -> Config:
    """Merge a list of config into a single config.

    Args:
        configs: List of configs.

    Returns:
        Config representation of the merged config objects.
    """
    return oc.OmegaConf.merge(*configs)


# %% CONVERTERS


def to_object(config: Config, resolve: bool = True) -> T.Dict[str, T.Any] | list | T.Any:
    """Convert a config object to a python object.

    Args:
        config: Representation of the config.
        resolve: Resolve variables. Defaults to True.

    Returns:
        Conversion of the config to a python object (dict, list, or primitive type).
    """
    return oc.OmegaConf.to_container(config, resolve=resolve)


# For backwards compatibility
def load_config(path: str) -> T.Dict[str, T.Any] | None:
    """Load configuration from YAML file and resolve environment variables.
    
    Args:
        path: Path to YAML config file
        
    Returns:
        Dict of configuration values with resolved environment variables
    """
    config = parse_file(path)
    if config is None:
        return None
    return T.cast(T.Dict[str, T.Any], to_object(config))
