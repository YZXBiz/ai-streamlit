"""Parse, merge, and convert configuration objects for file-based configs."""

# %% IMPORTS

import typing as T
from typing import Union

import omegaconf as oc

# %% TYPES

Config = Union[oc.ListConfig, oc.DictConfig]

# %% PARSERS


def parse_file(path: str) -> Config:
    """Parse a config file from a path.

    Args:
        path (str): path to local config.

    Returns:
        Config: representation of the config file.
    """
    return oc.OmegaConf.load(path)


def parse_string(string: str) -> Config:
    """Parse the given config string.

    Args:
        string (str): content of config string.

    Returns:
        Config: representation of the config string.
    """
    return oc.OmegaConf.create(string)


# For backward compatibility with existing code
def load_config(path: str) -> Config:
    """Load configuration from file.

    Args:
        path (str): path to configuration file.

    Returns:
        Config: loaded configuration.
    """
    return parse_file(path)


def save_config(config: Config, path: str) -> None:
    """Save configuration to file.

    Args:
        config (Config): configuration to save.
        path (str): path to save configuration.
    """
    oc.OmegaConf.save(config, path)


# %% MERGERS


def merge_configs(configs: T.Sequence[Config]) -> Config:
    """Merge a list of config into a single config.

    Args:
        configs (T.Sequence[Config]): list of configs.

    Returns:
        Config: representation of the merged config objects.
    """
    return oc.OmegaConf.merge(*configs)


# %% CONVERTERS


def to_object(config: Config, resolve: bool = True) -> object:
    """Convert a config object to a python object.

    Args:
        config (Config): representation of the config.
        resolve (bool): resolve variables. Defaults to True.

    Returns:
        object: conversion of the config to a python object.
    """
    return oc.OmegaConf.to_container(config, resolve=resolve)
